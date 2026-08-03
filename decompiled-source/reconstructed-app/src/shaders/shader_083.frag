, null);
  }
  init(tt) {
    Object.assign(this, tt);
    super.init();
    this.material = this.createRawShaderMaterial({
      uniforms: {
        u_texture: {
          value: null
        },
        u_vignetteFrom: {
          value: 0
        },
        u_vignetteTo: {
          value: 0
        },
        u_vignetteAspect: {
          value: this.vignetteAspect
        },
        u_vignetteColor: {
          value: this.vignetteColor
        },
        u_saturation: {
          value: 0
        },
        u_contrast: {
          value: 0
        },
        u_brightness: {
          value: 0
        },
        u_invert: {
          value: 0
        },
        u_tintColor: {
          value: this.tintColor
        },
        u_tintOpacity: {
          value: 0
        },
        u_bgColor: {
          value: this.bgColor
        },
        u_opacity: {
          value: 0
        },
        u_ditherSeed: {
          value: 0
        }
      },
      fragmentShader
    });
    this.cacheRenderTarget = fboHelper.createRenderTarget(1, 1);
    this.transitionMaterial = this.createRawShaderMaterial({
      uniforms: Object.assign({
        u_texture: {
          value: null
        },
        u_cacheTexture: {
          value: this.cacheRenderTarget.texture
        },
        u_transitionRatio: {
          value: 0
        },
        u_aspect: {
          value: new Vector2()
        }
      }, blueNoise.sharedUniforms),
      fragmentShader: transitionFragmentShader
    });
  }
  startTransition(tt) {
    let nt = tt.width;
    let rt = tt.height;
    this.cacheRenderTarget.setSize(nt, rt);
    fboHelper.copy(tt.fromRenderTarget.texture, this.cacheRenderTarget);
    tt.swap();
    this.transitionRatio = 0;
  }
  render(tt, nt = false) {
    const rt = tt.width;
    const it = tt.height;
    let ot = this.material.uniforms;
    ot.u_vignetteFrom.value = this.vignetteFrom;
    ot.u_vignetteTo.value = this.vignetteTo;
    const st = it / Math.sqrt(rt * rt + it * it);
    this.vignetteAspect.set(rt / it * st, st);
    ot.u_saturation.value = this.saturation - 1;
    ot.u_contrast.value = this.contrast;
    ot.u_brightness.value = this.brightness - 1;
    ot.u_invert.value = this.postInvert;
    ot.u_tintOpacity.value = this.tintOpacity;
    ot.u_opacity.value = this.opacity;
    ot.u_ditherSeed.value = Math.random() * 1000;
    let lt = this.transitionRatio;
    let ut = lt < 1;
    this.material.uniforms.u_texture.value = tt.fromTexture;
    tt.renderMaterial(this.material, tt.toRenderTarget);
    if (ut) {
      tt.swap();
      this.transitionMaterial.uniforms.u_texture.value = tt.fromTexture;
      this.transitionMaterial.uniforms.u_transitionRatio.value = lt;
      this.transitionMaterial.uniforms.u_aspect.value.copy(this.vignetteAspect);
      tt.renderMaterial(this.transitionMaterial, tt.toRenderTarget);
    }
    tt.swap();
    fboHelper.copy(tt.fromTexture, nt ? null : tt.toRenderTarget);
  }
}
var channelMixerFrag = `#define GLSLIFY 1
uniform sampler2D u_texture;uniform vec4 u_channelMixerR;uniform vec4 u_channelMixerG;uniform vec4 u_channelMixerB;uniform vec4 u_channelMixerA;varying vec2 v_uv;void main(){vec4 color=texture2D(u_texture,v_uv);gl_FragColor=vec4(dot(color,u_channelMixerR),dot(color,u_channelMixerG),dot(color,u_channelMixerB),dot(color,u_channelMixerA));}`;
class TextureHelper {
  constructor() {
    Et(this, 