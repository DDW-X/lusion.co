;
    this.referencePosition = new Vector3();
    this.nearDistance = 1;
    this.farDistance = 1000;
    this.map = null;
    this.alphaMap = null;
    this.displacementMap = null;
    this.displacementScale = 1;
    this.displacementBias = 0;
    this.fog = false;
    this.setValues($);
  }
  copy($) {
    super.copy($);
    this.referencePosition.copy($.referencePosition);
    this.nearDistance = $.nearDistance;
    this.farDistance = $.farDistance;
    this.map = $.map;
    this.alphaMap = $.alphaMap;
    this.displacementMap = $.displacementMap;
    this.displacementScale = $.displacementScale;
    this.displacementBias = $.displacementBias;
    return this;
  }
}
MeshDistanceMaterial.prototype.isMeshDistanceMaterial = true;
const vertex = `void main() {
	gl_Position = vec4( position, 1.0 );
}`;
const fragment = `uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
#include <packing>
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = unpackRGBATo2Half( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ) );
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = unpackRGBAToDepth( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ) );
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( squared_mean - mean * mean );
	gl_FragColor = pack2HalfToRGBA( vec2( mean, std_dev ) );
}`;
function WebGLShadowMap(et, $, tt) {
  let nt = new Frustum();
  const rt = new Vector2();
  const it = new Vector2();
  const ot = new Vector4();
  const st = new MeshDepthMaterial({
    depthPacking: RGBADepthPacking
  });
  const lt = new MeshDistanceMaterial();
  const ut = {};
  const ct = tt.maxTextureSize;
  const ht = {
    0: BackSide,
    1: FrontSide,
    2: DoubleSide
  };
  const ft = new ShaderMaterial({
    defines: {
      VSM_SAMPLES: 8
    },
    uniforms: {
      shadow_pass: {
        value: null
      },
      resolution: {
        value: new Vector2()
      },
      radius: {
        value: 4
      }
    },
    vertexShader: vertex,
    fragmentShader: fragment
  });
  const gt = ft.clone();
  gt.defines.HORIZONTAL_PASS = 1;
  const vt = new BufferGeometry();
  vt.setAttribute(