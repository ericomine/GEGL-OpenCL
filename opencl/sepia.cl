__kernel void cl_sepia(__global const float4 *in,
                       __global       float4 *out,
                                      float4 *coefs)
{
  int gid     = get_global_id(0);
  float4 in_v = in[gid];

  float4 aux0 = in_v * coefs[0];
  float4 aux1 = in_v * coefs[1];
  float4 aux2 = in_v * coefs[2];

  float4 out_v;
  out_v.x = aux0.x + aux0.y + aux0.z;
  out_v.y = aux1.x + aux1.y + aux1.z;
  out_v.z = aux2.x + aux2.y + aux2.z;
  out_v.w = in_v.w;

  out[gid] = out_v;
}
