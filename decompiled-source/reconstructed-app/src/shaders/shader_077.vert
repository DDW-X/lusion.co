)).render(<React.StrictMode><App /></React.StrictMode>);
}
var blitVert = `#define GLSLIFY 1
attribute vec2 position;varying vec2 v_uv;void main(){v_uv=position*0.5+0.5;gl_Position=vec4(position,0.0,1.0);}`;
var blitFrag = `#define GLSLIFY 1
uniform sampler2D u_texture;varying vec2 v_uv;void main(){gl_FragColor=texture2D(u_texture,v_uv);}`;
var copyAlpha = `#define GLSLIFY 1
uniform sampler2D u_textureFrom;uniform sampler2D u_textureTo;varying vec2 v_uv;void main(){vec4 from=texture2D(u_textureFrom,v_uv);vec4 to=texture2D(u_textureTo,v_uv);float alpha=from.a;gl_FragColor=vec4(to.rgb,alpha);}`;
var uvBlitVert = `#define GLSLIFY 1
attribute vec2 position;attribute vec2 uv;varying vec2 v_uv;void main(){v_uv=uv;gl_Position=vec4(position,0.0,1.0);}`;
var clearFrag = `#define GLSLIFY 1
uniform vec4 u_color;varying vec2 v_uv;void main(){gl_FragColor=u_color;}`;
var debugVert = `#define GLSLIFY 1
attribute vec3 position;attribute vec2 uv;uniform vec4 u_transform;varying vec2 v_uv;void main(){v_uv=uv;gl_Position=vec4(position.xy*u_transform.zw+u_transform.xy,0.0,1.0);}`;
class FboHelper {
  constructor() {
    Et(this, 