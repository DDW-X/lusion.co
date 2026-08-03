,
    uniforms: {
      envMap: {
        value: null
      },
      flipEnvMap: {
        value: -1
      }
    },
    vertexShader: _getCommonVertexShader(),
    fragmentShader: `

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,
    blending: NoBlending,
    depthTest: false,
    depthWrite: false
  });
}
function _getCommonVertexShader() {
  return `

		precision mediump float;
		precision mediump int;

		attribute float faceIndex;

		varying vec3 vOutputDirection;

		// RH coordinate system; PMREM face-indexing convention
		vec3 getDirection( vec2 uv, float face ) {

			uv = 2.0 * uv - 1.0;

			vec3 direction = vec3( uv, 1.0 );

			if ( face == 0.0 ) {

				direction = direction.zyx; // ( 1, v, u ) pos x

			} else if ( face == 1.0 ) {

				direction = direction.xzy;
				direction.xz *= -1.0; // ( -u, 1, -v ) pos y

			} else if ( face == 2.0 ) {

				direction.x *= -1.0; // ( -u, v, 1 ) pos z

			} else if ( face == 3.0 ) {

				direction = direction.zyx;
				direction.xz *= -1.0; // ( -1, v, -u ) neg x

			} else if ( face == 4.0 ) {

				direction = direction.xzy;
				direction.xy *= -1.0; // ( -u, -1, v ) neg y

			} else if ( face == 5.0 ) {

				direction.z *= -1.0; // ( u, v, -1 ) neg z

			}

			return direction;

		}

		void main() {

			vOutputDirection = getDirection( uv, faceIndex );
			gl_Position = vec4( position, 1.0 );

		}
	`;
}
function WebGLCubeUVMaps(et) {
  let $ = new WeakMap();
  let tt = null;
  function nt(st) {
    if (st && st.isTexture) {
      const lt = st.mapping;
      const ut = lt === EquirectangularReflectionMapping || lt === EquirectangularRefractionMapping;
      const ct = lt === CubeReflectionMapping || lt === CubeRefractionMapping;
      if (ut || ct) {
        if (st.isRenderTargetTexture && st.needsPMREMUpdate === true) {
          st.needsPMREMUpdate = false;
          let ht = $.get(st);
          if (tt === null) {
            tt = new PMREMGenerator(et);
          }
          ht = ut ? tt.fromEquirectangular(st, ht) : tt.fromCubemap(st, ht);
          $.set(st, ht);
          return ht.texture;
        } else {
          if ($.has(st)) {
            return $.get(st).texture;
          }
          {
            const ht = st.image;
            if (ut && ht && ht.height > 0 || ct && ht && rt(ht)) {
              if (tt === null) {
                tt = new PMREMGenerator(et);
              }
              const ft = ut ? tt.fromEquirectangular(st) : tt.fromCubemap(st);
              $.set(st, ft);
              st.addEventListener(