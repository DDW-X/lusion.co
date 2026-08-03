#define GLSLIFY 1
uniform sampler2D u_texture;uniform float u_luminosityThreshold;uniform float u_smoothWidth;
#ifdef USE_HALO
uniform vec2 u_texelSize;uniform vec2 u_aspect;uniform float u_haloWidth;uniform float u_haloRGBShift;uniform float u_haloStrength;uniform float u_haloMaskInner;uniform float u_haloMaskOuter;
#endif
varying vec2 v_uv;void main(){vec4 texel=texture2D(u_texture,v_uv);vec3 luma=vec3(0.299,0.587,0.114);float v=dot(texel.xyz,luma);vec4 outputColor=vec4(0.0,0.0,0.0,1.0);float alpha=smoothstep(u_luminosityThreshold,u_luminosityThreshold+u_smoothWidth,v);outputColor=mix(outputColor,texel,alpha);gl_FragColor=vec4(outputColor.rgb,1.0);
#ifdef USE_HALO
vec2 toCenter=(v_uv-0.5)*u_aspect;vec2 ghostUv=1.0-(toCenter+0.5);vec2 ghostVec=(vec2(0.5)-ghostUv);vec2 direction=normalize(ghostVec);vec2 haloVec=direction*u_haloWidth;float weight=length(vec2(0.5)-fract(ghostUv+haloVec));weight=pow(1.0-weight,3.0);vec3 distortion=vec3(-u_texelSize.x,0.0,u_texelSize.x)*u_haloRGBShift;vec2 uv=ghostUv+haloVec;gl_FragColor.rgb+=vec3(texture2D(u_texture,uv+direction*distortion.r).r,texture2D(u_texture,uv+direction*distortion.g).g,texture2D(u_texture,uv+direction*distortion.b).b)*u_haloStrength*smoothstep(u_haloMaskInner,u_haloMaskOuter,length(toCenter));
#endif
}