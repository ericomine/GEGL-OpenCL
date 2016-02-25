__kernel void cl_sepia(__global const float4 *in,
                       __global       float4 *out,
                                      float4 *c)
{
  int gid     = get_global_id(0);
  float4 in_v = in[gid];

  float4 out_v;
  out_v.x = dot(in_v, c[0]);
  out_v.y = dot(in_v, c[1]);
  out_v.z = dot(in_v, c[2]);
  out_v.w = in_v.w;
  out[gid] = out_v;
}
