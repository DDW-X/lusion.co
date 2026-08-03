);
    }
    this.material = fboHelper.createRawShaderMaterial({
      uniforms: Object.assign({
        u_texture: {
          value: null
        },
        u_screenPaintTexture: this.screenPaint.sharedUniforms.u_currPaintTexture,
        u_screenPaintTexelSize: this.screenPaint.sharedUniforms.u_paintTexelSize,
        u_amount: {
          value: 0
        },
        u_rgbShift: {
          value: 0
        },
        u_multiplier: {
          value: 0
        },
        u_colorMultiplier: {
          value: 0
        },
        u_shade: {
          value: 0
        }
      }, blueNoise.sharedUniforms),
      fragmentShader: frag$1
    });
  }
  needsRender(tt) {
    return this.amount > 0;
  }
  syncCamera(tt) {
    this.needsSync = true;
    if (tt) {
      tt.matrixWorldInverse.decompose(this._position, this._quaternion, this._scale);
      this.projectionViewMatrix.multiplyMatrices(tt.projectionMatrix, tt.matrixWorldInverse);
      this.projectionViewInverseMatrix.copy(this.projectionViewMatrix).invert();
    }
    this.prevProjectionViewMatrix.copy(this.projectionViewMatrix);
  }
  render(tt, nt = false) {
    this.material.uniforms.u_amount.value = this.amount;
    this.material.uniforms.u_rgbShift.value = this.rgbShift;
    this.material.uniforms.u_multiplier.value = this.multiplier;
    this.material.uniforms.u_colorMultiplier.value = this.colorMultiplier;
    this.material.uniforms.u_shade.value = this.shade;
    super.render(tt, nt);
  }
}
var fragmentShader = `#define GLSLIFY 1
varying vec2 v_uv;uniform sampler2D u_texture;uniform vec3 u_bgColor;uniform float u_opacity;uniform float u_vignetteFrom;uniform float u_vignetteTo;uniform vec2 u_vignetteAspect;uniform vec3 u_vignetteColor;uniform float u_saturation;uniform float u_contrast;uniform float u_brightness;uniform float u_invert;uniform vec3 u_tintColor;uniform float u_tintOpacity;uniform float u_ditherSeed;float hash13(vec3 p3){p3=fract(p3*.1031);p3+=dot(p3,p3.yzx+33.33);return fract((p3.x+p3.y)*p3.z);}vec3 screen(vec3 cb,vec3 cs){return cb+cs-(cb*cs);}vec3 colorDodge(vec3 cb,vec3 cs){return mix(min(vec3(1.0),cb/(1.0-cs)),vec3(1.0),step(vec3(1.0),cs));}void main(){vec2 uv=v_uv;vec3 color=texture2D(u_texture,uv).rgb;float luma=dot(color,vec3(0.299,0.587,0.114));color=mix(vec3(luma),color,1.0+u_saturation);color=0.5+(1.0+u_contrast)*(color-0.5);color+=u_brightness;color=mix(color,1.0-color,u_invert);color=mix(color,screen(colorDodge(color,u_tintColor),u_tintColor),u_tintOpacity);float d=length((uv-0.5)*u_vignetteAspect)*2.0;color=mix(color,u_vignetteColor,smoothstep(u_vignetteFrom,u_vignetteTo,d));gl_FragColor=vec4(mix(u_bgColor,color,u_opacity)+hash13(vec3(gl_FragCoord.xy,u_ditherSeed))/255.0,1.0);}`;
var transitionFragmentShader = `#define GLSLIFY 1
varying vec2 v_uv;uniform sampler2D u_texture;uniform sampler2D u_cacheTexture;uniform vec2 u_aspect;uniform float u_transitionRatio;void main(){vec2 uv=v_uv-0.5;float transitionRatioInverse=1.0-u_transitionRatio;uv*=u_aspect;float d=length(uv);float oriDist=d;float distortedDist=d*3.1415926*2.5*transitionRatioInverse;float distortion=cos(distortedDist*distortedDist);uv*=mix(1.0,distortion*distortion,u_transitionRatio);uv/=u_aspect;vec4 color=texture2D(u_texture,uv+0.5);vec4 sceneColor=texture2D(u_cacheTexture,uv*(1.0-u_transitionRatio*0.75)+0.5);float blend=smoothstep(0.4,0.6,transitionRatioInverse);color=mix(color,sceneColor,blend);gl_FragColor=color;}`;
class Final extends PostEffect {
  constructor() {
    super(...arguments);
    Et(this, 