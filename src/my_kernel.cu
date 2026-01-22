// private_vars: 各スレッド（ニューロン）の状態を保持するグローバルメモリ上の配列
// num_threads: 全体のスレッド数
extern "C"
#include <stdint.h>
// __constant__ int RSexci_param[34];
// __constant__ int RSinhi_param[34];
// __constant__ int FS_param[34];
// __constant__ int LTS_param[34];
// __constant__ int IB_param[34];
// __constant__ int EB_param[34];
// __constant__ int PB_param[34];
__constant__ int AllParams[7][34];
#define P(idx) p_base[idx]

__constant__ float dt;
__constant__ int buffer_size;
__constant__ int num_neurons;
__constant__ int num_synapses;
__constant__ float tr;
__constant__ float td;

__device__ int64_t v0(int64_t v, int64_t n, int64_t q, int64_t u, int64_t I, int64_t vv, const int* p_base);
__device__ int64_t n0(int64_t v, int64_t n, int64_t u, int64_t vv, const int* p_base);
__device__ int64_t q0(int64_t v, int64_t q, int64_t vv, const int* p_base);
__device__ int64_t u0(int64_t v, int64_t u, const int* p_base);

__global__ void update_neuron_state(
    int64_t* Vs_d, 
    int64_t* Ns_d, 
    int64_t* Qs_d, 
    int64_t* Us_d, 
    const unsigned char* neuron_type,
    // const int* RSexci_param,
    // const int* RSinhi_param,
    float* I_input,
    float* synaptic_input,
    unsigned char* last_spike, 
    unsigned char* raster,
    int current_step
) 
{
    // グローバルで一意なスレッドIDを計算
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // スレッド数が配列サイズを超えないようにガード
    if (tid < num_neurons) {
        // グローバルメモリからこのスレッドの以前の値を読み込む
        int64_t v = Vs_d[tid];
        int64_t n = Ns_d[tid];
        int64_t q = Qs_d[tid];
        int64_t u = Us_d[tid];

        int type_idx = neuron_type[tid];
        const int* p_base = AllParams[type_idx];

        /* [Fixed-Point Settings]
            0: BIT_Y_SHIFT          (Shift amount for normalization, usually 20)
            1: BIT_WIDTH_FRACTIONAL (Shift amount for inputs/thresholds, usually 10)
        */

        int64_t I = (int64_t)((synaptic_input[tid]+I_input[tid])*(1<<P(1)));
        int64_t vv = (int64_t)((v * v) / (1LL << P(1)));

        // PQN updates (compatible among all neuron types)
        // v update
        int64_t dv = v0(v, n, q, u, I, vv, p_base);
        // n update
        int64_t dn = n0(v, n, u, vv, p_base);
        // q update
        int64_t dq = q0(v, q, vv, p_base);
        // u update (enabled only when u params are set)
        int64_t du = u0(v, u, p_base);


        // 計算結果をグローバルメモリに書き戻して、次の呼び出しに備える
        Vs_d[tid] = v + dv;
        Ns_d[tid] = n + dn;
        Qs_d[tid] = q + dq;
        Us_d[tid] = u + du;

        int64_t threshold = (4 << P(1));
        unsigned char current_spike = (v > threshold) ? 1 : 0;
        raster[tid] = (current_spike && !last_spike[tid]);
        // last_spike[tid] = last_spike[tid] | raster[tid];
        last_spike[tid] = current_spike;
        synaptic_input[tid] = 0;
    }
}

__device__ int64_t v0(int64_t v, int64_t n, int64_t q, int64_t u, int64_t I, int64_t vv, const int* p_base) {
    /* [Membrane Potential (v) Dynamics]
        --- If v < 0 ---
        2: v_vv_S   (Coefficient for v^2)
        3: v_v_S    (Coefficient for v)
        4: v_c_S    (Constant term)
        --- If v >= 0 ---
        8: v_vv_L   (Coefficient for v^2)
        9: v_v_L    (Coefficient for v)
        10: v_c_L    (Constant term)
        --- Coupling Terms ---
        5: v_n      (Feedback from n)
        6: v_q      (Feedback from q)
        7: v_I      (Input current gain)
        33: v_u      (Feedback from u, used in PB mode. Is 0 for non-PB modes)
     */
    bool neg = (v < 0);
    int64_t c_vv = neg ? P(2) : P(8);
    int64_t c_v  = neg ? P(3) : P(9);
    int64_t c_c  = neg ? P(4) : P(10);
    
    return ((c_vv * vv) >> P(0)) +
           ((c_v  * v)  >> P(0)) +
            c_c +
           ((P(5) * n) >> P(0)) +
           ((P(6) * q) >> P(0)) +
           ((P(7) * I) >> P(0)) -
           ((P(33) * u) >> P(0));
}
__device__ int64_t n0(int64_t v, int64_t n, int64_t u, int64_t vv, const int* p_base) {
    /* [Recovery Variable (n) Dynamics]
        11: rg       (Threshold for n dynamics switching)
        --- If v < rg ---
        12: n_vv_S   (Coefficient for v^2)
        13: n_v_S    (Coefficient for v)
        14: n_c_S    (Constant term)
        --- If v >= rg ---
        16: n_vv_L   (Coefficient for v^2)
        17: n_v_L    (Coefficient for v)
        18: n_c_L    (Constant term)
        --- Self Decay ---
        15: n_n      (Decay rate of n)
        --- u-dependent scaling (only LTS/IB modes otherwise calculated eta to be 1) ---
        30: ru       (Threshold for u)
        31: n_uS     (Scaling factor eta0 if u < ru)
        32: n_uL     (Scaling factor eta1 if u >= ru)
    */
    bool cond = (v < P(11)); // rg
    int64_t c_vv = cond ? P(12) : P(16);
    int64_t c_v  = cond ? P(13) : P(17);
    int64_t c_c  = cond ? P(14) : P(18);

    int64_t dn = ((c_vv * vv) >> P(0)) +
                 ((c_v  * v)  >> P(0)) +
                    c_c +
                 ((P(15) * n) >> P(0));
    
    // IB/LTS: n scaling by u threshold (enabled only when eta params are set)
    int64_t eta = (u < (int64_t)P(30)) ? (int64_t)P(31) : (int64_t)P(32);
    return (dn * eta) >> P(0);
}
__device__ int64_t q0(int64_t v, int64_t q, int64_t vv, const int* p_base) {
    /* [Slow Variable (q) Dynamics]
        19: rh       (Threshold for q dynamics switching)
        --- If v < rh ---
        20: q_vv_S   (Coefficient for v^2)
        21: q_v_S    (Coefficient for v)
        22: q_c_S    (Constant term)
        --- If v >= rh ---
        24: q_vv_L   (Coefficient for v^2)
        25: q_v_L    (Coefficient for v)
        26: q_c_L    (Constant term)
        --- Self Decay ---
        23: q_q      (Decay rate of q)
    */
    bool cond = (v < P(19)); // rh
    int64_t c_vv = cond ? P(20) : P(24);
    int64_t c_v  = cond ? P(21) : P(25);
    int64_t c_c  = cond ? P(22) : P(26);

    return ((c_vv * vv) >> P(0)) +
           ((c_v  * v)  >> P(0)) +
            c_c +
           ((P(23) * q) >> P(0));
}
__device__ int64_t u0(int64_t v, int64_t u, const int* p_base) {
    /*[Ultra-Slow Variable (u) Dynamics] (LTS/IB/PB modes)
        27: u_v      (Coupling from v)
        28: u_u      (Self decay of u)
        29: u_c      (Constant term)
    */
    return (((int64_t)P(27) * v) >> P(0)) +
           (((int64_t)P(28) * u) >> P(0)) +
            (int64_t)P(29);
}

__global__ void copy_arrival_spike(
    unsigned char* arrival_spike,
    const unsigned char* delayed_spikes,
    int read_idx            //i%buffur_size
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_synapses){
        arrival_spike[tid] = delayed_spikes[read_idx*num_synapses + tid];
    }
}

__global__ void propagate_spikes(
    unsigned char* delayed_spikes,     // [in/out] 遅延バッファ
    const unsigned char* raster,             // [in] 全時間ステップのスパイク情報
    const unsigned char* spike_in,
    const int* which_neuron,
    const int* delay_row,             // [in] 遅延バッファの書き込み行
    int read_idx
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_synapses) {
        int neuron_id = which_neuron[tid];
        int delay = (delay_row[tid] + read_idx) % buffer_size;
        delayed_spikes[delay * num_synapses + tid] = raster[neuron_id] || spike_in[neuron_id];
    }
}

__global__ void synapses_calc(
    float* x,
    float* y,
    float* z,
    float* r,
    float* hr,
    const unsigned char* delayed_spikes,
    const unsigned char* mask_faci,
    const float* tau_rec,
    const float* tau_inact,
    const float* tau_faci,
    const float* U1,
    float* U,
    int read_idx
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_synapses) {
        float x_val = x[tid];
        float y_val = y[tid];
        float z_val = z[tid];
        float U_val = U[tid];
        unsigned char arrival = (delayed_spikes[read_idx*num_synapses + tid]);
        float tau_rec_val = tau_rec[tid];
        float tau_inact_val = tau_inact[tid];
        float r_val = r[tid];
        float hr_val = hr[tid];

        float dx = z_val / tau_rec_val * dt;
        float dy = -y_val / tau_inact_val * dt;
        float dz = (y_val / tau_inact_val - z_val / tau_rec_val) * dt;
        if (mask_faci[tid]) {
            float dU = -U_val / tau_faci[tid] * dt;
            dU += U1[tid] * (1.0f - U_val) * arrival;
            U_val += dU;
        }
        float temp = U_val * x_val * arrival;
        dx -= temp;
        dy += temp;
        r_val = r_val*(1.0f - dt / td) + hr_val*dt;
        hr_val = hr_val*(1.0f - dt / tr) + temp / (tr*td);
        
        x_val += dx;
        y_val += dy;
        z_val += dz;
        x[tid] = x_val;
        y[tid] = y_val;
        z[tid] = z_val;
        U[tid] = U_val;
        r[tid] = r_val;
        hr[tid] = hr_val;
    }
}

__global__ void mat_vec_mul(
    float* result,
    const int* neuron_to,
    const float* matrix,
    const float* vector
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_synapses) {
        float dot_product = 0.0f;
        int neuron_id = (int)neuron_to[tid];
        dot_product = matrix[tid] * vector[tid];
        atomicAdd(&result[neuron_id], dot_product);
    }
}
