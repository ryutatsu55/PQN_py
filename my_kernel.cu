// private_vars:
// 各スレッド（ニューロン）の状態を保持するグローバルメモリ上の配列 num_threads:
// 全体のスレッド数
extern "C"
#include <stdint.h>
__constant__ int PQN_param[34];
__constant__ float dt;
__constant__ int buffer_size;
__constant__ int num_neurons;
__constant__ int num_synapses;

__device__ int64_t v0(int64_t v, int64_t n, int64_t q, int64_t I, int64_t vv, const int* param);
__device__ int64_t n0(int64_t v, int64_t n, int64_t vv, const int* param);
__device__ int64_t q0(int64_t v, int64_t q, int64_t vv, const int* param);
__device__ int64_t u0(int64_t v, int64_t u, const int* param);

__global__ void update_neuron_state(
  int64_t* Vs_d,
  int64_t* Ns_d,
  int64_t* Qs_d,
  int64_t* Us_d, 
  float* I_input,
  float* synaptic_input,
  unsigned char* last_spike,
  unsigned char* raster,
  int current_step
) {
  // グローバルで一意なスレッドIDを計算
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // スレッド数が配列サイズを超えないようにガード
  if (tid < num_neurons) {
    // グローバルメモリからこのスレッドの以前の値を読み込む
    int64_t v = Vs_d[tid];
    int64_t n = Ns_d[tid];
    int64_t q = Qs_d[tid];
    int64_t u = Us_d[tid];

    const int* param = PQN_param;

    int64_t I =
        (int64_t)((synaptic_input[tid] + I_input[tid]) * (1 << param[1]));
    int64_t vv = (int64_t)((v * v) / (1LL << param[1]));

    // Base PQN updates (RS/FS/EB-compatible)
    int64_t dv = v0(v, n, q, I, vv, param);
    // PB: v-u coupling term (param[33] is 0 for non-PB modes)
    dv -= ((int64_t)param[33] * u) >> param[0];

    // n update
    int64_t dn = n0(v, n, vv, param);
    // IB/LTS: n scaling by u threshold (enabled only when eta params are set)
    if (param[31] != 0 || param[32] != 0) {
      if (u < (int64_t)param[30]) {
        dn = (dn * (int64_t)param[31]) >> param[0];
      } else {
        dn = (dn * (int64_t)param[32]) >> param[0];
      }
    }

    // q update
    int64_t dq = q0(v, q, vv, param);

    // u update (enabled only when u params are set)
    int64_t du = 0;
    if (param[27] != 0 || param[28] != 0 || param[29] != 0) {
      du = u0(v, u, param);
    }

    // 計算結果をグローバルメモリに書き戻して、次の呼び出しに備える
    Vs_d[tid] = v + dv;
    Ns_d[tid] = n + dn;
    Qs_d[tid] = q + dq;
    Us_d[tid] = u + du;

    int64_t threshold = (4 << param[1]);
    unsigned char current_spike = (v > threshold) ? 1 : 0;
    raster[tid] = (current_spike && !last_spike[tid]);
    // last_spike[tid] = last_spike[tid] | raster[tid];
    last_spike[tid] = current_spike;
    synaptic_input[tid] = 0;
  }
}

__device__ int64_t v0(int64_t v, int64_t n, int64_t q, int64_t I, int64_t vv, const int* param) {
  int64_t v0;
  if (v < 0) {
    v0 = ((param[2] * vv) >> param[0]) + ((param[3] * v) >> param[0]) +
         param[4] + ((param[5] * n) >> param[0]) +
         ((param[6] * q) >> param[0]) + ((param[7] * I) >> param[0]);
  } else {
    v0 = ((param[8] * vv) >> param[0]) + ((param[9] * v) >> param[0]) +
         param[10] + ((param[5] * n) >> param[0]) +
         ((param[6] * q) >> param[0]) + ((param[7] * I) >> param[0]);
  }
  return v0;
}
__device__ int64_t n0(int64_t v, int64_t n, int64_t vv, const int* param) {
  int64_t n0;
  if (v < param[11]) {
    n0 = ((param[12] * vv) >> param[0]) + ((param[13] * v) >> param[0]) +
         param[14] + ((param[15] * n) >> param[0]);
  } else {
    n0 = ((param[16] * vv) >> param[0]) + ((param[17] * v) >> param[0]) +
         param[18] + ((param[15] * n) >> param[0]);
  }
  return n0;
}
__device__ int64_t q0(int64_t v, int64_t q, int64_t vv, const int* param) {
  int64_t q0;
  if (v < param[19]) {
    q0 = ((param[20] * vv) >> param[0]) + ((param[21] * v) >> param[0]) +
         param[22] + ((param[23] * q) >> param[0]);
  } else {
    q0 = ((param[24] * vv) >> param[0]) + ((param[25] * v) >> param[0]) +
         param[26] + ((param[23] * q) >> param[0]);
  }
  return q0;
}
__device__ int64_t u0(int64_t v, int64_t u, const int* param) {
  // Indices (extended params):
  // 27: u_v, 28: u_u, 29: u_c
  int64_t du = 0;
  du += ((int64_t)param[27] * v) >> param[0];
  du += ((int64_t)param[28] * u) >> param[0];
  du += (int64_t)param[29];
  return du;
}

__global__ void copy_arrival_spike(
  unsigned char* arrival_spike,
  const unsigned char* delayed_spikes,
  int read_idx  // i%buffur_size
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_synapses) {
    arrival_spike[tid] = delayed_spikes[read_idx * num_synapses + tid];
  }
}

__global__ void propagate_spikes(
    unsigned char* delayed_spikes,  // [in/out] 遅延バッファ
    const unsigned char* raster,    // [in] 全時間ステップのスパイク情報
    const unsigned char* spike_in, 
    const int* which_neuron,
    const int* delay_row,  // [in] 遅延バッファの書き込み行
    int read_idx
  ) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_synapses) {
    int neuron_id = which_neuron[tid];
    int delay = (delay_row[tid] + read_idx) % buffer_size;
    delayed_spikes[delay * num_synapses + tid] =
        raster[neuron_id] || spike_in[neuron_id];
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
    float td, 
    float tr, 
    int read_idx
  ) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_synapses) {
    float x_val = x[tid];
    float y_val = y[tid];
    float z_val = z[tid];
    float U_val = U[tid];
    unsigned char arrival = (delayed_spikes[read_idx * num_synapses + tid]);
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
    r_val = r_val * (1.0f - dt / tr) + hr_val * dt;
    hr_val = hr_val * (1.0f - dt / td) + temp / (tr * td);

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
  ) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_synapses) {
    float dot_product = 0.0f;
    int neuron_id = (int)neuron_to[tid];
    dot_product = matrix[tid] * vector[tid];
    atomicAdd(&result[neuron_id], dot_product);
  }
}
