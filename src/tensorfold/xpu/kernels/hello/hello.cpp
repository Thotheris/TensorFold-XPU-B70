// Python bindings of the K0 smoke kernels (hello.sycl).

#include <torch/extension.h>

at::Tensor hello_add(const at::Tensor& a, const at::Tensor& b);
at::Tensor hello_sg_sum(const at::Tensor& x);
at::Tensor hello_dpas(const at::Tensor& a, const at::Tensor& b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &hello_add, "a + b, fp32");
    m.def("sg_sum", &hello_sg_sum, "sum over each sub-group of 16 by an xor butterfly");
    m.def("dpas", &hello_dpas, "8x16 bf16 times 16x16 bf16 into fp32 by one DPAS");
}
