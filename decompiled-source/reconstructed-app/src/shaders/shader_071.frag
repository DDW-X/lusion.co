);
      return;
    }
    this.renderTarget = nt;
    const rt = new PerspectiveCamera(fov, aspect, $, tt);
    rt.layers = this.layers;
    rt.up.set(0, -1, 0);
    rt.lookAt(new Vector3(1, 0, 0));
    this.add(rt);
    const it = new PerspectiveCamera(fov, aspect, $, tt);
    it.layers = this.layers;
    it.up.set(0, -1, 0);
    it.lookAt(new Vector3(-1, 0, 0));
    this.add(it);
    const ot = new PerspectiveCamera(fov, aspect, $, tt);
    ot.layers = this.layers;
    ot.up.set(0, 0, 1);
    ot.lookAt(new Vector3(0, 1, 0));
    this.add(ot);
    const st = new PerspectiveCamera(fov, aspect, $, tt);
    st.layers = this.layers;
    st.up.set(0, 0, -1);
    st.lookAt(new Vector3(0, -1, 0));
    this.add(st);
    const lt = new PerspectiveCamera(fov, aspect, $, tt);
    lt.layers = this.layers;
    lt.up.set(0, -1, 0);
    lt.lookAt(new Vector3(0, 0, 1));
    this.add(lt);
    const ut = new PerspectiveCamera(fov, aspect, $, tt);
    ut.layers = this.layers;
    ut.up.set(0, -1, 0);
    ut.lookAt(new Vector3(0, 0, -1));
    this.add(ut);
  }
  update($, tt) {
    if (this.parent === null) {
      this.updateMatrixWorld();
    }
    const nt = this.renderTarget;
    const [rt, it, ot, st, lt, ut] = this.children;
    const ct = $.xr.enabled;
    const ht = $.getRenderTarget();
    $.xr.enabled = false;
    const ft = nt.texture.generateMipmaps;
    nt.texture.generateMipmaps = false;
    $.setRenderTarget(nt, 0);
    $.render(tt, rt);
    $.setRenderTarget(nt, 1);
    $.render(tt, it);
    $.setRenderTarget(nt, 2);
    $.render(tt, ot);
    $.setRenderTarget(nt, 3);
    $.render(tt, st);
    $.setRenderTarget(nt, 4);
    $.render(tt, lt);
    nt.texture.generateMipmaps = ft;
    $.setRenderTarget(nt, 5);
    $.render(tt, ut);
    $.setRenderTarget(ht);
    $.xr.enabled = ct;
    nt.texture.needsPMREMUpdate = true;
  }
}
class CubeTexture extends Texture {
  constructor($, tt, nt, rt, it, ot, st, lt, ut, ct) {
    $ = $ !== undefined ? $ : [];
    tt = tt !== undefined ? tt : CubeReflectionMapping;
    super($, tt, nt, rt, it, ot, st, lt, ut, ct);
    this.flipY = false;
  }
  get images() {
    return this.image;
  }
  set images($) {
    this.image = $;
  }
}
CubeTexture.prototype.isCubeTexture = true;
class WebGLCubeRenderTarget extends WebGLRenderTarget {
  constructor($, tt = {}) {
    super($, $, tt);
    const nt = {
      width: $,
      height: $,
      depth: 1
    };
    const rt = [nt, nt, nt, nt, nt, nt];
    this.texture = new CubeTexture(rt, tt.mapping, tt.wrapS, tt.wrapT, tt.magFilter, tt.minFilter, tt.format, tt.type, tt.anisotropy, tt.encoding);
    this.texture.isRenderTargetTexture = true;
    this.texture.generateMipmaps = tt.generateMipmaps !== undefined ? tt.generateMipmaps : false;
    this.texture.minFilter = tt.minFilter !== undefined ? tt.minFilter : LinearFilter;
  }
  fromEquirectangularTexture($, tt) {
    this.texture.type = tt.type;
    this.texture.format = RGBAFormat;
    this.texture.encoding = tt.encoding;
    this.texture.generateMipmaps = tt.generateMipmaps;
    this.texture.minFilter = tt.minFilter;
    this.texture.magFilter = tt.magFilter;
    const nt = {
      uniforms: {
        tEquirect: {
          value: null
        }
      },
      vertexShader: `

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`,
      fragmentShader: `

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`
    };
    const rt = new BoxGeometry(5, 5, 5);
    const it = new ShaderMaterial({
      name: 