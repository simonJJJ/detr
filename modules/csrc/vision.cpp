#include "criss_cross_attention/ca.h"
#include "grid_attention/ga.h"
#include "grid_attention_align/ga_align.h"
#include "deformable_attention/ms_deform_attn.h"

namespace detr {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ca_forward", &ca_forward, "ca_forward");
    m.def("ca_backward", &ca_backward, "ca_backward");
    m.def("ca_map_forward", &ca_map_forward, "ca_map_forward");
    m.def("ca_map_backward", &ca_map_backward, "ca_map_backward");
    m.def("ga_forward", &ga_forward, "ga_forward");
    m.def("ga_backward", &ga_backward, "ga_backward");
    m.def("ga_map_forward", &ga_map_forward, "ga_map_forward");
    m.def("ga_map_backward", &ga_map_backward, "ga_map_backward");
    m.def("ga_align_forward", &ga_forward, "ga_forward");
    m.def("ga_align_backward", &ga_backward, "ga_backward");
    m.def("ga_map_align_forward", &ga_map_forward, "ga_map_forward");
    m.def("ga_map_align_backward", &ga_map_backward, "ga_map_backward");
    m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "ms_deform_attn_forward");
    m.def("ms_deform_attn_backward", &ms_deform_attn_backward, "ms_deform_attn_backward");
}

}  // namespace detr