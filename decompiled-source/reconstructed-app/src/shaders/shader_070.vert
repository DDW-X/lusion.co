, new Float32BufferAttribute(ht, 2));
    function vt(yt, wt, mt, _t, bt, Mt, St, At, Pt, Ot, Zt) {
      const Nt = Mt / Pt;
      const Tt = St / Ot;
      const Ft = Mt / 2;
      const $t = St / 2;
      const Wt = At / 2;
      const Kt = Pt + 1;
      const qt = Ot + 1;
      let jt = 0;
      let Bt = 0;
      const Xt = new Vector3();
      for (let Jt = 0; Jt < qt; Jt++) {
        const Lt = Jt * Tt - $t;
        for (let Rt = 0; Rt < Kt; Rt++) {
          const _n = Rt * Nt - Ft;
          Xt[yt] = _n * _t;
          Xt[wt] = Lt * bt;
          Xt[mt] = Wt;
          ut.push(Xt.x, Xt.y, Xt.z);
          Xt[yt] = 0;
          Xt[wt] = 0;
          Xt[mt] = At > 0 ? 1 : -1;
          ct.push(Xt.x, Xt.y, Xt.z);
          ht.push(Rt / Pt);
          ht.push(1 - Jt / Ot);
          jt += 1;
        }
      }
      for (let Jt = 0; Jt < Ot; Jt++) {
        for (let Lt = 0; Lt < Pt; Lt++) {
          const Rt = ft + Lt + Kt * Jt;
          const _n = ft + Lt + Kt * (Jt + 1);
          const Fn = ft + (Lt + 1) + Kt * (Jt + 1);
          const Vn = ft + (Lt + 1) + Kt * Jt;
          lt.push(Rt, _n, Vn);
          lt.push(_n, Fn, Vn);
          Bt += 6;
        }
      }
      st.addGroup(gt, Bt, Zt);
      gt += Bt;
      ft += jt;
    }
  }
  static fromJSON($) {
    return new BoxGeometry($.width, $.height, $.depth, $.widthSegments, $.heightSegments, $.depthSegments);
  }
}
function cloneUniforms(et) {
  const $ = {};
  for (const tt in et) {
    $[tt] = {};
    for (const nt in et[tt]) {
      const rt = et[tt][nt];
      if (rt && (rt.isColor || rt.isMatrix3 || rt.isMatrix4 || rt.isVector2 || rt.isVector3 || rt.isVector4 || rt.isTexture || rt.isQuaternion)) {
        $[tt][nt] = rt.clone();
      } else if (Array.isArray(rt)) {
        $[tt][nt] = rt.slice();
      } else {
        $[tt][nt] = rt;
      }
    }
  }
  return $;
}
function mergeUniforms(et) {
  const $ = {};
  for (let tt = 0; tt < et.length; tt++) {
    const nt = cloneUniforms(et[tt]);
    for (const rt in nt) {
      $[rt] = nt[rt];
    }
  }
  return $;
}
const UniformsUtils = {
  clone: cloneUniforms,
  merge: mergeUniforms
};
var default_vertex = `void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`;
var default_fragment = `void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`;
class ShaderMaterial extends Material {
  constructor($) {
    super();
    this.type = 