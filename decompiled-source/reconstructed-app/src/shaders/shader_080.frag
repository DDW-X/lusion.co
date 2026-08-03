);
    }
    const it = fboHelper.renderer;
    if (it) {
      it.autoClear = true;
      it.setClearColor(0, 0);
    }
    this.edgesMaterial.uniforms.u_texelSize = this.weightsMaterial.uniforms.u_texelSize = this.material.uniforms.u_texelSize = tt.sharedUniforms.u_texelSizeScaled;
    this.edgesMaterial.uniforms.u_texture.value = tt.sceneRenderTarget.texture;
    tt.renderMaterial(this.edgesMaterial, this.edgesRenderTarget);
    tt.renderMaterial(this.weightsMaterial, this.weightsRenderTarget);
    fboHelper.setColorState(rt);
    this.material.uniforms.u_texture.value = tt.sceneRenderTarget.texture;
    if (this.material.uniforms.u_texture) {
      this.material.uniforms.u_texture.value = tt.sceneRenderTarget.texture;
    }
    fboHelper.render(this.material, tt.tempSceneRenderTarget);
    this.fsr.render(tt.tempSceneRenderTarget.texture, tt.fromRenderTarget);
  }
  _createTexture(tt) {
    const nt = new Texture(tt);
    nt.generateMipmaps = false;
    nt.flipY = false;
    return nt;
  }
}
var fragmentShader$1 = `#define GLSLIFY 1
varying vec2 v_uv;uniform sampler2D u_texture;uniform sampler2D u_blurTexture0;
#if ITERATION > 1
uniform sampler2D u_blurTexture1;
#endif
#if ITERATION > 2
uniform sampler2D u_blurTexture2;
#endif
#if ITERATION > 3
uniform sampler2D u_blurTexture3;
#endif
#if ITERATION > 4
uniform sampler2D u_blurTexture4;
#endif
uniform float u_bloomWeights[ITERATION];
#include <common>
vec3 dithering(vec3 color){float grid_position=rand(gl_FragCoord.xy);vec3 dither_shift_RGB=vec3(0.25/255.0,-0.25/255.0,0.25/255.0);dither_shift_RGB=mix(2.0*dither_shift_RGB,-2.0*dither_shift_RGB,grid_position);return color+dither_shift_RGB;}void main(){vec4 c=texture2D(u_texture,v_uv);vec3 luma=vec3(0.299,0.587,0.114);float v=dot(c.xyz,luma);float a=1.0-v;gl_FragColor=c+(u_bloomWeights[0]*texture2D(u_blurTexture0,v_uv)
#if ITERATION > 1
+u_bloomWeights[1]*texture2D(u_blurTexture1,v_uv)
#endif
#if ITERATION > 2
+u_bloomWeights[2]*texture2D(u_blurTexture2,v_uv)
#endif
#if ITERATION > 3
+u_bloomWeights[3]*texture2D(u_blurTexture3,v_uv)
#endif
#if ITERATION > 4
+u_bloomWeights[4]*texture2D(u_blurTexture4,v_uv)
#endif
)*a;gl_FragColor.rgb=dithering(gl_FragColor.rgb);gl_FragColor.a=1.0;}`;
var highPassFragmentShader = `#define GLSLIFY 1
uniform sampler2D u_texture;uniform float u_luminosityThreshold;uniform float u_smoothWidth;
#ifdef USE_HALO
uniform vec2 u_texelSize;uniform vec2 u_aspect;uniform float u_haloWidth;uniform float u_haloRGBShift;uniform float u_haloStrength;uniform float u_haloMaskInner;uniform float u_haloMaskOuter;
#endif
varying vec2 v_uv;void main(){vec4 texel=texture2D(u_texture,v_uv);vec3 luma=vec3(0.299,0.587,0.114);float v=dot(texel.xyz,luma);vec4 outputColor=vec4(0.0,0.0,0.0,1.0);float alpha=smoothstep(u_luminosityThreshold,u_luminosityThreshold+u_smoothWidth,v);outputColor=mix(outputColor,texel,alpha);gl_FragColor=vec4(outputColor.rgb,1.0);
#ifdef USE_HALO
vec2 toCenter=(v_uv-0.5)*u_aspect;vec2 ghostUv=1.0-(toCenter+0.5);vec2 ghostVec=(vec2(0.5)-ghostUv);vec2 direction=normalize(ghostVec);vec2 haloVec=direction*u_haloWidth;float weight=length(vec2(0.5)-fract(ghostUv+haloVec));weight=pow(1.0-weight,3.0);vec3 distortion=vec3(-u_texelSize.x,0.0,u_texelSize.x)*u_haloRGBShift;vec2 uv=ghostUv+haloVec;gl_FragColor.rgb+=vec3(texture2D(u_texture,uv+direction*distortion.r).r,texture2D(u_texture,uv+direction*distortion.g).g,texture2D(u_texture,uv+direction*distortion.b).b)*u_haloStrength*smoothstep(u_haloMaskInner,u_haloMaskOuter,length(toCenter));
#endif
}`;
var blurFragmentShader = `#define GLSLIFY 1
varying vec2 v_uv;uniform sampler2D u_texture;uniform vec2 u_resolution;uniform vec2 u_direction;float gaussianPdf(in float x,in float sigma){return 0.39894*exp(-0.5*x*x/(sigma*sigma))/sigma;}void main(){vec2 invSize=1.0/u_resolution;float fSigma=float(SIGMA);float weightSum=gaussianPdf(0.0,fSigma);vec3 diffuseSum=texture2D(u_texture,v_uv).rgb*weightSum;for(int i=1;i<KERNEL_RADIUS;i++){float x=float(i);float w=gaussianPdf(x,fSigma);vec2 uvOffset=u_direction*invSize*x;vec3 sample1=texture2D(u_texture,v_uv+uvOffset).rgb;vec3 sample2=texture2D(u_texture,v_uv-uvOffset).rgb;diffuseSum+=(sample1+sample2)*w;weightSum+=2.0*w;}gl_FragColor=vec4(diffuseSum/weightSum,1.0);}`;
class Bloom extends PostEffect {
  constructor() {
    super(...arguments);
    Et(this, 