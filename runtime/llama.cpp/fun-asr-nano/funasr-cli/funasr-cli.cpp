// funasr-cli: end-to-end Fun-ASR-Nano in C++ on the llama.cpp / ggml stack.
//
//   wav(16k mono) -> kaldi fbank -> SAN-M encoder + adaptor (ggml) ->
//   low-frame-rate truncation -> [prefix tokens | audio embeds | suffix tokens]
//   -> Qwen3 LLM (llama.cpp) -> transcription.
//
// This is the whisper.cpp-style single-binary path: no Python at runtime.
//
//   funasr-cli --enc funasr-encoder.gguf -m qwen3-0.6b.gguf -a audio.wav

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// any audio (wav/mp3/flac, any rate/channels) -> 16 kHz mono f32, via miniaudio
#define FUNASR_AUDIO_IMPLEMENTATION
#include "funasr_audio.h"
// built-in FSMN-VAD front end (single-binary --vad segmentation)
#include "funasr_vad.h"
#include "funasr_srt.h"
#include <utility>

// ======================= kaldi fbank + LFR =======================
static const int FS=16000, WINLEN=400, SHIFT=160, NFFT=512, NMEL=80, LFR_M=7, LFR_N=6;
static const float PREEMPH=0.97f, LOWF=20.0f, HIGHF=8000.0f;
static inline float mel(float f){ return 1127.0f*logf(1.0f+f/700.0f); }
static void fft(std::vector<float>&re,std::vector<float>&im,int n){
    for(int i=1,j=0;i<n;i++){int b=n>>1;for(;j&b;b>>=1)j^=b;j^=b;if(i<j){std::swap(re[i],re[j]);std::swap(im[i],im[j]);}}
    for(int len=2;len<=n;len<<=1){double a=-2.0*M_PI/len;float wr=cosf(a),wi=sinf(a);
        for(int i=0;i<n;i+=len){float cr=1,ci=0;for(int k=0;k<len/2;k++){
            float ur=re[i+k],ui=im[i+k];float vr=re[i+k+len/2]*cr-im[i+k+len/2]*ci,vi=re[i+k+len/2]*ci+im[i+k+len/2]*cr;
            re[i+k]=ur+vr;im[i+k]=ui+vi;re[i+k+len/2]=ur-vr;im[i+k+len/2]=ui-vi;
            float n2=cr*wr-ci*wi;ci=cr*wi+ci*wr;cr=n2;}}}
}
// returns [T x 560] row-major, sets T
static std::vector<float> compute_fbank(std::vector<float> wav, int & T_out) {
    for (auto & v : wav) v *= 32768.0f;
    std::vector<float> win(WINLEN);
    for (int i=0;i<WINLEN;i++) win[i]=0.54f-0.46f*cosf(2.0f*M_PI*i/(WINLEN-1));
    const int NBIN=NFFT/2+1; float bw=(float)FS/NFFT, ml=mel(LOWF), mh=mel(HIGHF), dm=(mh-ml)/(NMEL+1);
    std::vector<std::vector<float>> fb(NMEL, std::vector<float>(NBIN,0.0f));
    for(int m=0;m<NMEL;m++){float L=ml+m*dm,C=ml+(m+1)*dm,R=ml+(m+2)*dm;
        for(int k=0;k<NBIN;k++){float mf=mel(bw*k); if(mf>L&&mf<R) fb[m][k]=mf<=C?(mf-L)/(C-L):(R-mf)/(R-C);}}
    int N=wav.size(); int T=(N-WINLEN)/SHIFT+1;
    std::vector<std::vector<float>> feat(T, std::vector<float>(NMEL));
    std::vector<float> re(NFFT),im(NFFT),fr(WINLEN);
    const float fl=1.1920929e-07f;
    for(int t=0;t<T;t++){const float*s=wav.data()+t*SHIFT;
        double mn=0;for(int i=0;i<WINLEN;i++)mn+=s[i];mn/=WINLEN;
        for(int i=0;i<WINLEN;i++)fr[i]=s[i]-(float)mn;
        for(int i=WINLEN-1;i>0;i--)fr[i]-=PREEMPH*fr[i-1];fr[0]-=PREEMPH*fr[0];
        for(int i=0;i<NFFT;i++){re[i]=i<WINLEN?fr[i]*win[i]:0.0f;im[i]=0.0f;}
        fft(re,im,NFFT);
        for(int m=0;m<NMEL;m++){float e=0;for(int k=0;k<NBIN;k++)if(fb[m][k]>0)e+=fb[m][k]*(re[k]*re[k]+im[k]*im[k]);
            feat[t][m]=logf(e>fl?e:fl);}}
    // LFR
    const int pad=(LFR_M-1)/2; int T_lfr=(T+LFR_N-1)/LFR_N;
    std::vector<std::vector<float>> pd; pd.reserve(T+pad+LFR_M);
    for(int i=0;i<pad;i++)pd.push_back(feat[0]);
    for(int t=0;t<T;t++)pd.push_back(feat[t]);
    while((int)pd.size()<(T_lfr-1)*LFR_N+LFR_M)pd.push_back(feat[T-1]);
    int D=LFR_M*NMEL; std::vector<float> out((size_t)T_lfr*D);
    for(int i=0;i<T_lfr;i++)for(int j=0;j<LFR_M;j++)
        memcpy(&out[(size_t)i*D+j*NMEL],pd[i*LFR_N+j].data(),NMEL*sizeof(float));
    T_out=T_lfr; return out;
}

// ======================= ggml SAN-M encoder + adaptor =======================
struct cfg { int d_model=512,n_head=4,num_blocks=50,tp_blocks=20,kernel=11,adp_llm=1024,adp_layers=2,adp_head=8; };
struct enc_model { cfg c; ggml_context*ctx_w=nullptr; std::map<std::string,ggml_tensor*> t;
    ggml_tensor* g(const std::string&n){auto it=t.find(n);if(it==t.end()){fprintf(stderr,"missing %s\n",n.c_str());exit(1);}return it->second;} };
static const float LN_EPS=1e-5f;
static bool load_enc(const char*p, enc_model&m){
    gguf_init_params gp={false,&m.ctx_w}; gguf_context*g=gguf_init_from_file(p,gp); if(!g)return false;
    auto rd=[&](const char*k,int d){int i=gguf_find_key(g,k);return i<0?d:(int)gguf_get_val_u32(g,i);};
    m.c.d_model=rd("funasr.enc.output_size",512); m.c.n_head=rd("funasr.enc.attention_heads",4);
    m.c.num_blocks=rd("funasr.enc.num_blocks",50); m.c.tp_blocks=rd("funasr.enc.tp_blocks",20);
    m.c.kernel=rd("funasr.enc.kernel_size",11); m.c.adp_llm=rd("funasr.adp.llm_dim",1024);
    m.c.adp_layers=rd("funasr.adp.n_layer",2); m.c.adp_head=rd("funasr.adp.attention_heads",8);
    int n=gguf_get_n_tensors(g); for(int i=0;i<n;i++){const char*nm=gguf_get_tensor_name(g,i);m.t[nm]=ggml_get_tensor(m.ctx_w,nm);}
    gguf_free(g); return true;
}
static ggml_tensor* lin(ggml_context*c,ggml_tensor*w,ggml_tensor*b,ggml_tensor*x){auto y=ggml_mul_mat(c,w,x);return b?ggml_add(c,y,b):y;}
static ggml_tensor* lnorm(ggml_context*c,ggml_tensor*x,ggml_tensor*g,ggml_tensor*b){return ggml_add(c,ggml_mul(c,ggml_norm(c,x,LN_EPS),g),b);}
static ggml_tensor* sanm_attn(ggml_context*c,enc_model&m,const std::string&p,ggml_tensor*x,int T){
    const int D=m.c.d_model,H=m.c.n_head,dk=D/H,K=m.c.kernel;
    ggml_tensor*qkv=lin(c,m.g(p+"linear_q_k_v.weight"),m.g(p+"linear_q_k_v.bias"),x); size_t nb1=qkv->nb[1];
    ggml_tensor*q=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,0));
    ggml_tensor*k=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,(size_t)D*sizeof(float)));
    ggml_tensor*v=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,(size_t)2*D*sizeof(float)));
    const int pad=(K-1)/2; ggml_tensor*fk=m.g(p+"fsmn_block.weight");
    ggml_tensor*vp=ggml_pad_ext(c,v,0,0,pad,pad,0,0,0,0); ggml_tensor*fsmn=v;
    for(int j=0;j<K;j++){auto sl=ggml_view_2d(c,vp,D,T,vp->nb[1],(size_t)j*vp->nb[1]);
        auto wj=ggml_view_1d(c,fk,D,(size_t)j*fk->nb[1]); fsmn=ggml_add(c,fsmn,ggml_mul(c,ggml_cont(c,sl),wj));}
    q=ggml_permute(c,ggml_reshape_3d(c,q,dk,H,T),0,2,1,3); k=ggml_permute(c,ggml_reshape_3d(c,k,dk,H,T),0,2,1,3);
    ggml_tensor*vh=ggml_cont(c,ggml_permute(c,ggml_reshape_3d(c,v,dk,H,T),1,2,0,3));
    ggml_tensor*kq=ggml_soft_max(c,ggml_scale(c,ggml_mul_mat(c,k,q),1.0f/sqrtf((float)dk)));
    ggml_tensor*o=ggml_cont_2d(c,ggml_permute(c,ggml_mul_mat(c,vh,kq),0,2,1,3),D,T);
    return ggml_add(c,lin(c,m.g(p+"linear_out.weight"),m.g(p+"linear_out.bias"),o),fsmn);
}
static ggml_tensor* sanm_layer(ggml_context*c,enc_model&m,const std::string&p,ggml_tensor*x,int T,bool res){
    auto r=x; auto h=lnorm(c,x,m.g(p+"norm1.weight"),m.g(p+"norm1.bias"));
    auto sa=sanm_attn(c,m,p+"self_attn.",h,T); x=res?ggml_add(c,r,sa):sa; r=x;
    h=lnorm(c,x,m.g(p+"norm2.weight"),m.g(p+"norm2.bias"));
    h=lin(c,m.g(p+"feed_forward.w_1.weight"),m.g(p+"feed_forward.w_1.bias"),h); h=ggml_relu(c,h);
    h=lin(c,m.g(p+"feed_forward.w_2.weight"),m.g(p+"feed_forward.w_2.bias"),h); return ggml_add(c,r,h);
}
static ggml_tensor* adp_layer(ggml_context*c,enc_model&m,const std::string&p,ggml_tensor*x,int T){
    const int D=m.c.adp_llm,H=m.c.adp_head,dk=D/H; auto r=x;
    auto h=lnorm(c,x,m.g(p+"norm1.weight"),m.g(p+"norm1.bias"));
    auto q=ggml_permute(c,ggml_reshape_3d(c,lin(c,m.g(p+"self_attn.linear_q.weight"),m.g(p+"self_attn.linear_q.bias"),h),dk,H,T),0,2,1,3);
    auto k=ggml_permute(c,ggml_reshape_3d(c,lin(c,m.g(p+"self_attn.linear_k.weight"),m.g(p+"self_attn.linear_k.bias"),h),dk,H,T),0,2,1,3);
    auto vh=ggml_cont(c,ggml_permute(c,ggml_reshape_3d(c,lin(c,m.g(p+"self_attn.linear_v.weight"),m.g(p+"self_attn.linear_v.bias"),h),dk,H,T),1,2,0,3));
    auto kq=ggml_soft_max(c,ggml_scale(c,ggml_mul_mat(c,k,q),1.0f/sqrtf((float)dk)));
    auto o=ggml_cont_2d(c,ggml_permute(c,ggml_mul_mat(c,vh,kq),0,2,1,3),D,T);
    x=ggml_add(c,r,lin(c,m.g(p+"self_attn.linear_out.weight"),m.g(p+"self_attn.linear_out.bias"),o)); r=x;
    h=lnorm(c,x,m.g(p+"norm2.weight"),m.g(p+"norm2.bias"));
    h=lin(c,m.g(p+"feed_forward.w_1.weight"),m.g(p+"feed_forward.w_1.bias"),h); h=ggml_relu(c,h);
    h=lin(c,m.g(p+"feed_forward.w_2.weight"),m.g(p+"feed_forward.w_2.bias"),h); return ggml_add(c,r,h);
}
static void add_posenc(std::vector<float>&x,int T,int depth){
    double inc=log(10000.0)/(depth/2.0-1.0);
    for(int t=0;t<T;t++){double pos=t+1;for(int i=0;i<depth/2;i++){double its=exp(i*-inc),st=pos*its;
        x[(size_t)t*depth+i]+=(float)sin(st);x[(size_t)t*depth+depth/2+i]+=(float)cos(st);}}
}
// fbank [T x F] -> adaptor out [T x adp_llm] row-major
static std::vector<float> run_encoder(enc_model&m,std::vector<float> fbank,int T,int F,int&Dout){
    float sc=sqrtf((float)m.c.d_model); for(auto&v:fbank)v*=sc; add_posenc(fbank,T,F);
    ggml_backend_t be=ggml_backend_cpu_init();
    ggml_init_params cp={(size_t)1024*1024*1024,nullptr,true}; ggml_context*c=ggml_init(cp);
    ggml_tensor*inp=ggml_new_tensor_2d(c,GGML_TYPE_F32,F,T); ggml_set_input(inp);
    ggml_tensor*x=sanm_layer(c,m,"audio_encoder.encoders0.0.",inp,T,false);
    for(int i=0;i<m.c.num_blocks-1;i++) x=sanm_layer(c,m,"audio_encoder.encoders."+std::to_string(i)+".",x,T,true);
    x=lnorm(c,x,m.g("audio_encoder.after_norm.weight"),m.g("audio_encoder.after_norm.bias"));
    for(int i=0;i<m.c.tp_blocks;i++) x=sanm_layer(c,m,"audio_encoder.tp_encoders."+std::to_string(i)+".",x,T,true);
    x=lnorm(c,x,m.g("audio_encoder.tp_norm.weight"),m.g("audio_encoder.tp_norm.bias"));
    x=lin(c,m.g("audio_adaptor.linear1.weight"),m.g("audio_adaptor.linear1.bias"),x); x=ggml_relu(c,x);
    x=lin(c,m.g("audio_adaptor.linear2.weight"),m.g("audio_adaptor.linear2.bias"),x);
    for(int i=0;i<m.c.adp_layers;i++) x=adp_layer(c,m,"audio_adaptor.blocks."+std::to_string(i)+".",x,T);
    ggml_set_output(x);
    ggml_cgraph*gf=ggml_new_graph_custom(c,32768,false); ggml_build_forward_expand(gf,x);
    ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_cpu_buffer_type()); ggml_gallocr_alloc_graph(ga,gf);
    ggml_backend_tensor_set(inp,fbank.data(),0,ggml_nbytes(inp));
    ggml_backend_cpu_set_n_threads(be,8); ggml_backend_graph_compute(be,gf);
    Dout=(int)x->ne[0]; std::vector<float> out((size_t)Dout*T); ggml_backend_tensor_get(x,out.data(),0,ggml_nbytes(x));
    ggml_gallocr_free(ga); ggml_free(c); ggml_backend_free(be); return out;
}

// ======================= LLM (llama.cpp) =======================
static int decode_batch(llama_context*ctx,int n,llama_token*tok,float*embd,int n_embd,int&n_past,bool last_logits){
    std::vector<llama_pos> pos(n); std::vector<int32_t> nsid(n,1);
    std::vector<llama_seq_id> s0(1,0); std::vector<llama_seq_id*> sid(n); std::vector<int8_t> lg(n,0);
    for(int i=0;i<n;i++){pos[i]=n_past+i;sid[i]=s0.data();}
    if(last_logits) lg[n-1]=1;
    llama_batch b={n,tok,embd,pos.data(),nsid.data(),sid.data(),lg.data()};
    int r=llama_decode(ctx,b); n_past+=n; return r;
}

// ======================= prompt / model (shared by CLI and --server) =======================
static const char* SYS_PREFIX =
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n";
static const char* CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n";

// "满意 方便" / "满意\u3000方便" -> {"满意","方便"}; same input shape as the
// Python client sends in the optional `hotword` field.
static std::vector<std::string> split_hotwords(const std::string& s){
    std::vector<std::string> out; std::string cur; size_t i=0;
    while(i<s.size()){
        unsigned char c=(unsigned char)s[i];
        bool ascii_ws = (c==' '||c=='\t'||c=='\r'||c=='\n');
        bool ideographic_space = (c==0xE3 && i+2<s.size() &&
                                  (unsigned char)s[i+1]==0x80 && (unsigned char)s[i+2]==0x80);
        if(ascii_ws || ideographic_space){
            if(!cur.empty()){ out.push_back(cur); cur.clear(); }
            i += ideographic_space ? 3 : 1;
        } else { cur.push_back(s[i]); i++; }
    }
    if(!cur.empty()) out.push_back(cur);
    return out;
}

// Mirrors funasr/models/fun_asr_nano/inference_vllm.py::_build_prompt_text()
// so the C++ server biases the LLM with the same wording as the Python stack.
static std::string build_user_text(const std::string& hotword){
    auto hw = split_hotwords(hotword);
    if(hw.empty()) return "语音转写：";
    std::string joined;
    for(size_t i=0;i<hw.size();i++){ if(i) joined += ", "; joined += hw[i]; }
    std::string p = "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n";
    p += "热词列表：[" + joined + "]\n";
    p += "语音转写：";
    return p;
}

static std::vector<llama_token> tokenize_text(const llama_vocab* v,const std::string& s){
    int n=-llama_tokenize(v,s.c_str(),(int)s.size(),nullptr,0,false,true);
    if(n<=0) return {};
    std::vector<llama_token> out(n);
    llama_tokenize(v,s.c_str(),(int)s.size(),out.data(),n,false,true);
    return out;
}

struct asr_model { enc_model em; llama_model* model=nullptr; const llama_vocab* vocab=nullptr; };

static bool load_asr_model(asr_model& m,const std::string& enc_path,const std::string& llm_path){
    if(!load_enc(enc_path.c_str(),m.em)) return false;
    ggml_backend_load_all();
    llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=0;
    m.model=llama_model_load_from_file(llm_path.c_str(),mp);
    if(!m.model) return false;
    m.vocab=llama_model_get_vocab(m.model);
    return true;
}

static void free_asr_model(asr_model& m){
    if(m.model) llama_model_free(m.model);
    if(m.em.ctx_w) ggml_free(m.em.ctx_w);
    m.model=nullptr; m.vocab=nullptr; m.em.ctx_w=nullptr;
}

static llama_context* new_llama_ctx(asr_model& m){
    llama_context_params cp=llama_context_default_params();
    cp.n_ctx=2048; cp.n_batch=2048; cp.n_ubatch=2048;
    return llama_init_from_model(m.model,cp);
}

static llama_sampler* new_sampler(const llama_vocab* vocab,float rep){
    auto sp=llama_sampler_chain_default_params();
    llama_sampler* smpl=llama_sampler_chain_init(sp);
    if(rep!=1.0f) llama_sampler_chain_add(smpl,llama_sampler_init_penalties(llama_vocab_n_tokens(vocab),256,rep,0.0f,0.0f));
    llama_sampler_chain_add(smpl,llama_sampler_init_greedy());
    return smpl;
}

// ======================= one request =======================
struct asr_opts {
    int npred=512;
    double chunk_sec=0;
    float rep=1.0f;
    int vad_maxseg=30000;
    bool log_vad=true;          // CLI parity: log segment count; server keeps stderr quiet
    std::string vad_path;
};

struct seg_result { int start_ms=0; int end_ms=0; std::string text; };

// Same decode path as the CLI's single-file mode, but reuse a resident
// llama_context. The caller owns ctx; VAD/fbank/encoder work is per request.
static bool transcribe_segments(asr_model& m,llama_context* ctx,const asr_opts& o,
                                const std::string& wav_path,const std::string& hotword,
                                std::vector<seg_result>& segs){
    std::vector<float> wav;
    if(!funasr_load_audio_16k_mono(wav_path.c_str(),wav)) return false;

    std::vector<std::pair<int,int>> wins;
    if(!o.vad_path.empty()){
        std::vector<std::pair<int,int>> vsegs; // ms
        if(!funasr_vad_segments(o.vad_path,wav,o.vad_maxseg,vsegs)) return false;
        for(auto&s:vsegs){
            int off=(int)((int64_t)s.first*16000/1000), end=(int)((int64_t)s.second*16000/1000);
            if(end>(int)wav.size()) end=wav.size();
            if(end-off>0) wins.push_back({off,end-off});
        }
        if(o.log_vad) fprintf(stderr,"[vad] %zu segments\n",wins.size());
    } else {
        int chunk_n = o.chunk_sec > 0 ? std::max(1,(int)(o.chunk_sec*16000)) : (int)wav.size();
        for(size_t off=0; off<wav.size(); off+=chunk_n)
            wins.push_back({(int)off,(int)std::min((size_t)chunk_n,wav.size()-off)});
    }

    std::vector<llama_token> pre = tokenize_text(m.vocab, std::string(SYS_PREFIX)+build_user_text(hotword));
    std::vector<llama_token> suf = tokenize_text(m.vocab, CHAT_SUFFIX);
    if(pre.empty() || suf.empty()) return false;

    llama_sampler* smpl = new_sampler(m.vocab,o.rep);
    for(auto& w:wins){
        int off=w.first, len=w.second;
        if(len < WINLEN) continue;                    // too short for one frame
        int start_ms=(int)((int64_t)off*1000/16000);
        int end_ms=(int)((int64_t)(off+len)*1000/16000);
        std::vector<float> seg(wav.begin()+off,wav.begin()+off+len);
        int T=0; auto fbank=compute_fbank(seg,T);
        int D=0; auto adp=run_encoder(m.em,fbank,T,560,D);
        int ol=1+(T-3+2)/2; ol=1+(ol-3+2)/2; int n_aud=(ol-1)/2+1;

        llama_memory_clear(llama_get_memory(ctx),true);  // fresh context per window
        int n_past=0;
        decode_batch(ctx,(int)pre.size(),pre.data(),nullptr,0,n_past,false);
        decode_batch(ctx,n_aud,nullptr,adp.data(),D,n_past,false);
        decode_batch(ctx,(int)suf.size(),suf.data(),nullptr,0,n_past,true);

        std::string text;
        llama_token tk=llama_sampler_sample(smpl,ctx,-1);
        for(int i=0;i<o.npred;i++){
            if(llama_vocab_is_eog(m.vocab,tk)) break;
            char buf[256]; int k=llama_token_to_piece(m.vocab,tk,buf,sizeof(buf),0,true);
            if(k>0) text.append(buf,k);
            decode_batch(ctx,1,&tk,nullptr,0,n_past,true);
            tk=llama_sampler_sample(smpl,ctx,-1);
        }
        segs.push_back({start_ms,end_ms,text});
    }
    llama_sampler_free(smpl);
    return true;
}

// ======================= minimal JSON (stdin/stdout protocol) =======================
struct json_request { bool has_id=false; long long id=0; std::string audio; std::string hotword; };

static void json_skip_ws(const std::string& s,size_t& i){
    while(i<s.size() && (s[i]==' '||s[i]=='\t'||s[i]=='\r'||s[i]=='\n')) i++;
}

static bool json_hex4(const std::string& s,size_t& i,unsigned& cp){
    if(i+4>s.size()) return false; cp=0;
    for(int k=0;k<4;k++){
        char h=s[i++]; cp<<=4;
        if(h>='0'&&h<='9') cp|=(unsigned)(h-'0');
        else if(h>='a'&&h<='f') cp|=(unsigned)(h-'a'+10);
        else if(h>='A'&&h<='F') cp|=(unsigned)(h-'A'+10);
        else return false;
    }
    return true;
}

static void utf8_append(std::string& out,unsigned cp){
    if(cp<0x80) out.push_back((char)cp);
    else if(cp<0x800){ out.push_back((char)(0xC0|(cp>>6))); out.push_back((char)(0x80|(cp&0x3F))); }
    else if(cp<0x10000){
        out.push_back((char)(0xE0|(cp>>12)));
        out.push_back((char)(0x80|((cp>>6)&0x3F)));
        out.push_back((char)(0x80|(cp&0x3F)));
    } else {
        out.push_back((char)(0xF0|(cp>>18)));
        out.push_back((char)(0x80|((cp>>12)&0x3F)));
        out.push_back((char)(0x80|((cp>>6)&0x3F)));
        out.push_back((char)(0x80|(cp&0x3F)));
    }
}

static bool json_parse_string(const std::string& s,size_t& i,std::string& out){
    if(i>=s.size() || s[i]!='"') return false;
    i++; out.clear();
    while(i<s.size()){
        char c=s[i++];
        if(c=='"') return true;
        if(c!='\\'){ out.push_back(c); continue; }
        if(i>=s.size()) return false;
        char e=s[i++];
        switch(e){
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                unsigned cp=0;
                if(!json_hex4(s,i,cp)) return false;
                if(cp>=0xD800 && cp<=0xDBFF){   // surrogate pair
                    size_t j=i;
                    if(j+6<=s.size() && s[j]=='\\' && s[j+1]=='u'){
                        j+=2; unsigned lo=0;
                        if(json_hex4(s,j,lo) && lo>=0xDC00 && lo<=0xDFFF){
                            cp=0x10000+((cp-0xD800)<<10)+(lo-0xDC00);
                            i=j;
                        }
                    }
                }
                utf8_append(out,cp);
                break;
            }
            default: return false;
        }
    }
    return false;
}

static bool json_skip_value(const std::string& s,size_t& i){
    json_skip_ws(s,i);
    if(i>=s.size()) return false;
    char c=s[i];
    if(c=='"'){ std::string tmp; return json_parse_string(s,i,tmp); }
    if(c=='{' || c=='['){
        char close = c=='{' ? '}' : ']';
        i++; json_skip_ws(s,i);
        if(i<s.size() && s[i]==close){ i++; return true; }
        while(true){
            if(c=='{'){
                std::string key; json_skip_ws(s,i);
                if(!json_parse_string(s,i,key)) return false;
                json_skip_ws(s,i);
                if(i>=s.size() || s[i]!=':') return false;
                i++;
            }
            if(!json_skip_value(s,i)) return false;
            json_skip_ws(s,i);
            if(i>=s.size()) return false;
            if(s[i]==','){ i++; continue; }
            if(s[i]==close){ i++; return true; }
            return false;
        }
    }
    static const char* lits[] = { "true", "false", "null" };
    for(auto lit : lits){
        size_t n=strlen(lit);
        if(s.compare(i,n,lit)==0){ i+=n; return true; }
    }
    if(c=='-' || (c>='0'&&c<='9')){
        i++;
        while(i<s.size() && (s[i]=='.'||s[i]=='e'||s[i]=='E'||s[i]=='+'||s[i]=='-'||(s[i]>='0'&&s[i]<='9')) i++;
        return true;
    }
    return false;
}

static bool parse_request(const std::string& s,json_request& r){
    size_t i=0; json_skip_ws(s,i);
    if(i>=s.size() || s[i]!='{') return false;
    i++; json_skip_ws(s,i);
    if(i<s.size() && s[i]=='}') return true;
    while(i<s.size()){
        std::string key;
        if(!json_parse_string(s,i,key)) return false;
        json_skip_ws(s,i);
        if(i>=s.size() || s[i]!=':') return false;
        i++; json_skip_ws(s,i);
        if(key=="id"){
            if(i<s.size() && s[i]=='"'){
                std::string v;
                if(!json_parse_string(s,i,v)) return false;
                r.id=strtoll(v.c_str(),nullptr,10); r.has_id=true;
            } else if(i<s.size() && (s[i]=='-' || (s[i]>='0'&&s[i]<='9'))){
                const char* start=s.c_str()+i; char* end=nullptr;
                long long v=strtoll(start,&end,10);
                if(end==start) return false;
                i+=(size_t)(end-start); r.id=v; r.has_id=true;
            } else return false;
        } else if(key=="audio"){
            if(!json_parse_string(s,i,r.audio)) return false;
        } else if(key=="hotword"){
            if(!json_parse_string(s,i,r.hotword)) return false;
        } else {
            if(!json_skip_value(s,i)) return false;
        }
        json_skip_ws(s,i);
        if(i>=s.size()) return false;
        if(s[i]=='}'){ i++; return true; }
        if(s[i]!=',') return false;
        i++; json_skip_ws(s,i);
    }
    return false;
}

static std::string json_escape(const std::string& s){
    std::string o; o.reserve(s.size()+8);
    for(unsigned char c : s){
        switch(c){
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            case '\b': o += "\\b"; break;
            case '\f': o += "\\f"; break;
            default:
                if(c<0x20){ char b[8]; snprintf(b,sizeof(b),"\\u%04x",(unsigned)c); o += b; }
                else o.push_back((char)c);
        }
    }
    return o;
}

// ======================= resident server =======================
// Protocol (see docs/llamacpp-server-protocol.md in the asr-stream service):
//   stdin : one JSON per line  {"id":1,"audio":"C:\\tmp\\seg.wav","hotword":"满意 方便"}
//   stdout: {"ready": true} once, then {"id":1,"text":"...","ms":123} or {"id":1,"error":"..."}
// Model/encoder are loaded once; each worker owns one llama_context so requests
// from different ids can be decoded concurrently.
static int run_server(asr_model& m,const asr_opts& o,int n_workers){
    if(n_workers<1) n_workers=1;
    if(n_workers>16) n_workers=16;
    std::vector<llama_context*> ctxs;
    for(int i=0;i<n_workers;i++){
        llama_context* ctx=new_llama_ctx(m);
        if(!ctx){
            fprintf(stderr,"[server] failed to create llama context (%d/%d)\n",i+1,n_workers);
            for(auto c:ctxs) llama_free(c);
            return 1;
        }
        ctxs.push_back(ctx);
    }

    fprintf(stderr,"[server] ready, workers=%d\n",n_workers);
    printf("{\"ready\": true}\n");
    fflush(stdout);

    std::mutex mtask, mout;
    std::condition_variable cv;
    std::deque<std::string> tasks;
    bool stop=false;

    auto respond=[&](const std::string& payload){
        std::lock_guard<std::mutex> lock(mout);
        fwrite(payload.data(),1,payload.size(),stdout);
        fflush(stdout);
    };

    auto handle=[&](llama_context* ctx,const std::string& line){
        json_request req;
        if(!parse_request(line,req) || !req.has_id){
            respond("{\"error\":\"bad json\"}\n");
            return;
        }
        int64_t t0=ggml_time_us();
        std::vector<seg_result> segs;
        bool ok=false;
        try {
            ok=transcribe_segments(m,ctx,o,req.audio,req.hotword,segs);
        } catch(const std::exception& e){
            fprintf(stderr,"[server] id=%lld exception: %s\n",req.id,e.what());
        } catch(...){
            fprintf(stderr,"[server] id=%lld exception\n",req.id);
        }
        int64_t t1=ggml_time_us();
        std::string out;
        if(!ok){
            out="{\"id\":"+std::to_string(req.id)+",\"error\":\"failed to transcribe audio\"}\n";
        } else {
            std::string text;
            for(auto& s : segs) text += s.text;
            out="{\"id\":"+std::to_string(req.id)+",\"text\":\""+json_escape(text)+
                "\",\"ms\":"+std::to_string((t1-t0)/1000)+"}\n";
        }
        respond(out);
    };

    std::vector<std::thread> workers;
    for(int i=0;i<n_workers;i++){
        llama_context* ctx=ctxs[i];
        workers.emplace_back([&,ctx](){
            while(true){
                std::string line;
                {
                    std::unique_lock<std::mutex> lock(mtask);
                    cv.wait(lock,[&]{ return stop || !tasks.empty(); });
                    if(tasks.empty()) break;   // only when stop && empty
                    line=tasks.front(); tasks.pop_front();
                }
                handle(ctx,line);
            }
        });
    }

    std::string line;
    while(std::getline(std::cin,line)){
        while(!line.empty() && (line.back()=='\r' || line.back()=='\n')) line.pop_back();
        if(line.empty()) continue;
        {
            std::lock_guard<std::mutex> lock(mtask);
            tasks.push_back(line);
        }
        cv.notify_one();
    }
    {
        std::lock_guard<std::mutex> lock(mtask);
        stop=true;
    }
    cv.notify_all();
    for(auto& t : workers) t.join();
    for(auto c : ctxs) llama_free(c);
    fprintf(stderr,"[server] stdin closed, exit\n");
    return 0;
}

static void print_usage(const char*argv0){
    fprintf(stderr,
        "usage: %s --enc enc.gguf -m llm.gguf -a audio.wav"
        " [-n npred] [--chunk sec]"
        " [--vad fsmn-vad.gguf [--vad-maxseg ms]]"
        " [--srt] [--rep R]\n"
        "       %s --server [--server-workers N] --enc enc.gguf -m llm.gguf\n"
        "  (--server: resident process, one JSON request per stdin line;"
        " model loads once)\n", argv0, argv0);
}

int main(int argc,char**argv){
    std::string enc_path,llm_path,wav_path,vad_path;
    int npred=512; double chunk_sec=0; float rep=1.0f; int vad_maxseg=30000;
    bool srt_mode=false; bool server_mode=false; int server_workers=1;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--enc")&&i+1<argc)enc_path=argv[++i];
        else if(!strcmp(argv[i],"-m")&&i+1<argc)llm_path=argv[++i];
        else if(!strcmp(argv[i],"-a")&&i+1<argc)wav_path=argv[++i];
        else if(!strcmp(argv[i],"-n")&&i+1<argc)npred=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--chunk")&&i+1<argc)chunk_sec=atof(argv[++i]);
        else if(!strcmp(argv[i],"--vad")&&i+1<argc)vad_path=argv[++i];
        else if(!strcmp(argv[i],"--vad-maxseg")&&i+1<argc)vad_maxseg=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--srt"))srt_mode=true;
        else if(!strcmp(argv[i],"--rep")&&i+1<argc)rep=atof(argv[++i]);
        else if(!strcmp(argv[i],"--server"))server_mode=true;
        else if(!strcmp(argv[i],"--server-workers")&&i+1<argc)server_workers=atoi(argv[++i]);
        else { print_usage(argv[0]); return 1; }
    }
    if(enc_path.empty()||llm_path.empty()||(!server_mode&&wav_path.empty())){
        fprintf(stderr,"missing args\n");print_usage(argv[0]);return 1;
    }

    ggml_time_init(); // Required before ggml_time_us() on Windows.

    asr_model m;
    if(!load_asr_model(m,enc_path,llm_path)) return 1;

    asr_opts o; o.npred=npred; o.chunk_sec=chunk_sec; o.rep=rep;
    o.vad_maxseg=vad_maxseg; o.vad_path=vad_path; o.log_vad=!server_mode;

    if(server_mode){
        int rc=run_server(m,o,server_workers);
        free_asr_model(m);
        return rc;
    }

    int64_t t0=ggml_time_us();
    llama_context* ctx=new_llama_ctx(m);
    if(!ctx){fprintf(stderr,"failed to create llama context\n");free_asr_model(m);return 1;}

    std::vector<seg_result> segs;
    if(!transcribe_segments(m,ctx,o,wav_path,"",segs)){
        fprintf(stderr,"failed to transcribe audio\n");
        llama_free(ctx); free_asr_model(m); return 1;
    }

    int srt_idx=0;
    for(auto& s : segs){
        if(srt_mode){
            if(s.text != "/sil"){
                srt_idx++;
                format_srt_line(srt_idx,s.start_ms,s.end_ms,s.text);
            }
        } else {
            printf("%s",s.text.c_str());
        }
        fflush(stdout);
    }
    if(!srt_mode) printf("\n");
    int64_t t2=ggml_time_us();
    fprintf(stderr,"[done] %.2fs ; chunk=%.0fs\n",(t2-t0)/1e6,chunk_sec);
    llama_free(ctx); free_asr_model(m);
    return 0;
}
