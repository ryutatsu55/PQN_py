// private_vars: 各スレッド（ニューロン）の状態を保持するグローバルメモリ上の配列
// num_threads: 全体のスレッド数
extern "C"
__constant__ int const_param[27];
__global__ void update_neuron_state(
    int* Vs_d, 
    int* Ns_d, 
    int* Qs_d, 
    float* I_float, 
    float* synaptic_input,
    unsigned char* last_spike, 
    unsigned char* raster, 
    int num_threads, 
    int current_step
) 
{
    // グローバルで一意なスレッドIDを計算
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // スレッド数が配列サイズを超えないようにガード
    if (tid < num_threads) {
        // グローバルメモリからこのスレッドの以前の値を読み込む
        int v = Vs_d[tid];
        int n = Ns_d[tid];
        int q = Qs_d[tid];

        int I = (int)((I_float[current_step * num_threads + tid]+synaptic_input[tid])*(1<<const_param[1]));
        int vv = (int)(((long long)v * v) / (1LL << const_param[1]));
        int v0;
        if(v < 0){
            v0 =(((long long)const_param[2] * vv) >> const_param[0]) +
                (((long long)const_param[4] * v) >> const_param[0]) +
                const_param[6] +
                (((long long)const_param[8] * n) >> const_param[0]) +
                (((long long)const_param[9] * q) >> const_param[0]) +
                (((long long)const_param[10] * I) >> const_param[0]);
        }else{
            v0 =(((long long)const_param[3] * vv) >> const_param[0]) +
                (((long long)const_param[5] * v) >> const_param[0]) +
                const_param[7] +
                (((long long)const_param[8] * n) >> const_param[0]) +
                (((long long)const_param[9] * q) >> const_param[0]) +
                (((long long)const_param[10] * I) >> const_param[0]);
        }
        v += v0;

        int n0;
        if(v<const_param[18]){
            n0 =(((long long)const_param[11] * vv) >> const_param[0]) +
                (((long long)const_param[13] * v) >> const_param[0]) +
                const_param[16] +
                (((long long)const_param[15] * n) >> const_param[0]);
        }else{
            n0 =(((long long)const_param[12] * vv) >> const_param[0]) +
                (((long long)const_param[14] * v) >> const_param[0]) +
                const_param[17] +
                (((long long)const_param[15] * n) >> const_param[0]);
        }
        n += n0;

        int q0;
        if(v<const_param[26]){
            q0 =(((long long)const_param[19] * vv) >> const_param[0]) +
                (((long long)const_param[21] * v) >> const_param[0]) +
                const_param[24] +
                (((long long)const_param[23] * q) >> const_param[0]);
        }else{
            q0 =(((long long)const_param[20] * vv) >> const_param[0]) +
                (((long long)const_param[22] * v) >> const_param[0]) +
                const_param[25] +
                (((long long)const_param[23] * q) >> const_param[0]);
        }
        q += q0;
        

        // 計算結果をグローバルメモリに書き戻して、次の呼び出しに備える
        Vs_d[tid] = v;
        Ns_d[tid] = n;
        Qs_d[tid] = q;

        int threshold = (4 << const_param[1]);
        unsigned char current_spike = (v > threshold) ? 1 : 0;
        raster[tid] = (current_spike && !last_spike[tid]);
        // last_spike[tid] = last_spike[tid] | raster[tid];
        last_spike[tid] = current_spike;
    }
}

__global__ void copy_arrival_spike(
    unsigned char* arrival_spike,
    const unsigned char* delayed_spikes,
    int read_idx,            //i%buffur_size
    int num_synapses
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
    const int* which_neuron,
    const int* delay_row,             // [in] 遅延バッファの書き込み行
    int num_synapses,          // [in] ニューロンあたりのシナプス数
    int read_idx,
    int buffur_size
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_synapses) {
        int neuron_id = which_neuron[tid];
        int delay = (delay_row[tid] + read_idx) % buffur_size;
        delayed_spikes[delay * num_synapses + tid] = raster[neuron_id];
    }
}

__global__ void synapses_calc(
    float* x,
    float* y,
    float* z,
    const unsigned char* arrival_spike,
    const unsigned char* mask_faci,
    const float* tau_rec,
    const float* tau_inact,
    const float* tau_faci,
    const float* U1,
    float* U,
    int num_synapses,
    float dt
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_synapses) {
        float x_val = x[tid];
        float y_val = y[tid];
        float z_val = z[tid];
        float U_val = U[tid];
        unsigned char arrival = arrival_spike[tid];
        float tau_rec_val = tau_rec[tid];
        float tau_inact_val = tau_inact[tid];

        float dx = z_val / tau_rec_val * dt;
        float dy = -y_val / tau_inact_val * dt;
        float dz = (y_val / tau_inact_val - z_val / tau_rec_val) * dt;
        if (mask_faci[tid]) {
            float dU = -U_val / tau_faci[tid] * dt;
            if (arrival) {
                dU += U1[tid] * (1.0f - U_val);
            }
            U_val += dU;
        }
        if (arrival) {
            dx -= U_val * x_val;
            dy += U_val * x_val;
        }
        x_val += dx;
        y_val += dy;
        z_val += dz;
        x[tid] = x_val;
        y[tid] = y_val;
        z[tid] = z_val;
        U[tid] = U_val;
    }
}

__global__ void mat_vec_mul(
    float* result,
    const float* matrix,
    const float* vector,
    int rows,
    int cols
)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        float dot_product = 0.0f;
        // 内積を計算
        for (int col = 0; col < cols; ++col) {
            dot_product += matrix[row * cols + col] * vector[col];
        }
        result[row] = dot_product;
    }
}
