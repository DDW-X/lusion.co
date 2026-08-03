, 0, 3, 0.00001);
    }
  }
  _add($, tt, nt, rt, it) {
    this.addInput($, tt, {
      min: nt,
      max: rt,
      step: it
    });
  }
}
var gui = new GUI$1();
var particlesVert = `#define GLSLIFY 1
attribute vec4 a_random;attribute vec2 a_simUv;uniform sampler2D u_prevPosTex;uniform sampler2D u_currPosTex;uniform vec2 u_resolution;uniform float u_time;uniform float u_opacity;uniform float u_pSizeMul;uniform float u_pSoftMul;uniform float u_focusDist;varying float v_softness;varying float v_opacity;varying vec3 v_color;float sizeFromLife(float life){float cut=0.008;return(1.0-smoothstep(1.0-cut,1.0,life))*smoothstep(0.0,cut,life);}void main(){vec4 positionLife=texture2D(u_currPosTex,a_simUv);float lifeSize=sizeFromLife(positionLife.w);vec3 pos=positionLife.xyz;vec4 mvPosition=modelViewMatrix*vec4(pos,1.0);float dist=u_focusDist*10.0;float coef=abs(-mvPosition.z-dist)*0.3+pow(max(0.0,-mvPosition.z-dist),2.5)*0.5;v_softness=coef*u_pSoftMul*10.0;v_opacity=1.0;v_opacity=u_opacity*lifeSize;gl_Position=projectionMatrix*mvPosition;float pSize=(coef*200.0*u_pSizeMul)/-mvPosition.z*u_resolution.y/1280.0;gl_PointSize=pSize*lifeSize;}`;
var particlesFrag = `#define GLSLIFY 1
varying float v_softness;varying float v_opacity;varying vec3 v_color;float linearStep(float edge0,float edge1,float x){return clamp((x-edge0)/(edge1-edge0),0.0,1.0);}void main(){float d=length(gl_PointCoord.xy*2.0-1.0);float b=linearStep(0.0,v_softness+fwidth(d),1.0-d);vec3 color=v_color*b*v_opacity;gl_FragColor=vec4(color,b*v_opacity);}`;
var particlePositionShader = `#define GLSLIFY 1
uniform sampler2D u_defaultPosTex;uniform sampler2D u_prevPosTex;uniform sampler2D u_currVelTex;uniform sampler2D u_logoPosTex;uniform float u_simDieSpeed;uniform vec3 u_curlNoiseScale;uniform vec3 u_curlStrength;uniform float u_curlStrMul;uniform float u_simSpeed;uniform vec3 u_bounds;uniform vec3 u_screenBounds;uniform float u_deltaTime;uniform float u_time;uniform float u_mode;uniform float u_logoCutPercent;varying vec2 v_uv;vec4 mod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}float mod289(float x){return x-floor(x*(1.0/289.0))*289.0;}vec4 permute(vec4 x){return mod289(((x*34.0)+1.0)*x);}float permute(float x){return mod289(((x*34.0)+1.0)*x);}vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}float taylorInvSqrt(float r){return 1.79284291400159-0.85373472095314*r;}vec4 grad4(float j,vec4 ip){const vec4 ones=vec4(1.0,1.0,1.0,-1.0);vec4 p,s;p.xyz=floor(fract(vec3(j)*ip.xyz)*7.0)*ip.z-1.0;p.w=1.5-dot(abs(p.xyz),ones.xyz);s=vec4(lessThan(p,vec4(0.0)));p.xyz=p.xyz+(s.xyz*2.0-1.0)*s.www;return p;}
#define F4 0.309016994374947451
vec4 simplexNoiseDerivatives(vec4 v){const vec4 C=vec4(0.138196601125011,0.276393202250021,0.414589803375032,-0.447213595499958);vec4 i=floor(v+dot(v,vec4(F4)));vec4 x0=v-i+dot(i,C.xxxx);vec4 i0;vec3 isX=step(x0.yzw,x0.xxx);vec3 isYZ=step(x0.zww,x0.yyz);i0.x=isX.x+isX.y+isX.z;i0.yzw=1.0-isX;i0.y+=isYZ.x+isYZ.y;i0.zw+=1.0-isYZ.xy;i0.z+=isYZ.z;i0.w+=1.0-isYZ.z;vec4 i3=clamp(i0,0.0,1.0);vec4 i2=clamp(i0-1.0,0.0,1.0);vec4 i1=clamp(i0-2.0,0.0,1.0);vec4 x1=x0-i1+C.xxxx;vec4 x2=x0-i2+C.yyyy;vec4 x3=x0-i3+C.zzzz;vec4 x4=x0+C.wwww;i=mod289(i);float j0=permute(permute(permute(permute(i.w)+i.z)+i.y)+i.x);vec4 j1=permute(permute(permute(permute(i.w+vec4(i1.w,i2.w,i3.w,1.0))+i.z+vec4(i1.z,i2.z,i3.z,1.0))+i.y+vec4(i1.y,i2.y,i3.y,1.0))+i.x+vec4(i1.x,i2.x,i3.x,1.0));vec4 ip=vec4(1.0/294.0,1.0/49.0,1.0/7.0,0.0);vec4 p0=grad4(j0,ip);vec4 p1=grad4(j1.x,ip);vec4 p2=grad4(j1.y,ip);vec4 p3=grad4(j1.z,ip);vec4 p4=grad4(j1.w,ip);vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));p0*=norm.x;p1*=norm.y;p2*=norm.z;p3*=norm.w;p4*=taylorInvSqrt(dot(p4,p4));vec3 values0=vec3(dot(p0,x0),dot(p1,x1),dot(p2,x2));vec2 values1=vec2(dot(p3,x3),dot(p4,x4));vec3 m0=max(0.5-vec3(dot(x0,x0),dot(x1,x1),dot(x2,x2)),0.0);vec2 m1=max(0.5-vec2(dot(x3,x3),dot(x4,x4)),0.0);vec3 temp0=-6.0*m0*m0*values0;vec2 temp1=-6.0*m1*m1*values1;vec3 mmm0=m0*m0*m0;vec2 mmm1=m1*m1*m1;float dx=temp0[0]*x0.x+temp0[1]*x1.x+temp0[2]*x2.x+temp1[0]*x3.x+temp1[1]*x4.x+mmm0[0]*p0.x+mmm0[1]*p1.x+mmm0[2]*p2.x+mmm1[0]*p3.x+mmm1[1]*p4.x;float dy=temp0[0]*x0.y+temp0[1]*x1.y+temp0[2]*x2.y+temp1[0]*x3.y+temp1[1]*x4.y+mmm0[0]*p0.y+mmm0[1]*p1.y+mmm0[2]*p2.y+mmm1[0]*p3.y+mmm1[1]*p4.y;float dz=temp0[0]*x0.z+temp0[1]*x1.z+temp0[2]*x2.z+temp1[0]*x3.z+temp1[1]*x4.z+mmm0[0]*p0.z+mmm0[1]*p1.z+mmm0[2]*p2.z+mmm1[0]*p3.z+mmm1[1]*p4.z;float dw=temp0[0]*x0.w+temp0[1]*x1.w+temp0[2]*x2.w+temp1[0]*x3.w+temp1[1]*x4.w+mmm0[0]*p0.w+mmm0[1]*p1.w+mmm0[2]*p2.w+mmm1[0]*p3.w+mmm1[1]*p4.w;return vec4(dx,dy,dz,dw)*49.0;}vec3 curl(in vec3 p,in float noiseTime,in float persistence){vec4 xNoisePotentialDerivatives=vec4(0.0);vec4 yNoisePotentialDerivatives=vec4(0.0);vec4 zNoisePotentialDerivatives=vec4(0.0);for(int i=0;i<2;++i){float twoPowI=pow(2.0,float(i));float scale=0.5*twoPowI*pow(persistence,float(i));xNoisePotentialDerivatives+=simplexNoiseDerivatives(vec4(p*twoPowI,noiseTime))*scale;yNoisePotentialDerivatives+=simplexNoiseDerivatives(vec4((p+vec3(123.4,129845.6,-1239.1))*twoPowI,noiseTime))*scale;zNoisePotentialDerivatives+=simplexNoiseDerivatives(vec4((p+vec3(-9519.0,9051.0,-123.0))*twoPowI,noiseTime))*scale;}return vec3(zNoisePotentialDerivatives[1]-yNoisePotentialDerivatives[2],xNoisePotentialDerivatives[2]-zNoisePotentialDerivatives[0],yNoisePotentialDerivatives[0]-xNoisePotentialDerivatives[1]);}vec3 hash33(vec3 p3){p3=fract(p3*vec3(.1031,.1030,.0973));p3+=dot(p3,p3.yxz+33.33);return fract((p3.xxy+p3.yxx)*p3.zyx);}void main(){vec4 positionLife=texture2D(u_prevPosTex,v_uv);vec4 velInfo=texture2D(u_currVelTex,v_uv);positionLife.w-=u_deltaTime*u_simDieSpeed*0.01*(1.0+velInfo.w);if(positionLife.w<0.0){vec3 h=hash33(vec3(v_uv,u_time));float modeCut=step(u_logoCutPercent,h.x);if(u_mode*modeCut>0.5){vec3 p=texture2D(u_logoPosTex,v_uv).xyz;positionLife.xyz=p+h*0.2;}else{positionLife.xyz=texture2D(u_defaultPosTex,v_uv).xyz;}positionLife.w=1.0;}vec3 boundCheck=step(positionLife.xyz,u_bounds);boundCheck*=step(-u_bounds,positionLife.xyz);positionLife.w*=boundCheck.x*boundCheck.y*boundCheck.z;positionLife.xyz+=velInfo.xyz*u_deltaTime;vec3 curlStr=u_curlStrength*u_curlStrMul;vec3 curlScale=u_curlNoiseScale*1.0;vec3 curlVel=curl(positionLife.xyz*curlScale,u_time*u_simSpeed,0.02)*curlStr*u_deltaTime;curlVel/=1.0+velInfo.w*u_mode;positionLife.xyz+=curlVel;gl_FragColor=positionLife;}`;
var particleVelocityShader = `#define GLSLIFY 1
uniform sampler2D u_prevPosTex;uniform sampler2D u_prevVelTex;uniform sampler2D u_logoPosTex;uniform sampler2D u_mousePaintTex;uniform float u_attractForce;uniform vec3 u_windForce;uniform float u_windStrMul;uniform float u_mouseMoveIntensity;uniform float u_mouseStrength;uniform vec3 u_screenBounds;uniform float u_simDieSpeed;uniform float u_deltaTime;uniform float u_time;uniform float u_mode;uniform float u_logoCutPercent;varying vec2 v_uv;vec3 hash33(vec3 p3){p3=fract(p3*vec3(.1031,.1030,.0973));p3+=dot(p3,p3.yxz+33.33);return fract((p3.xxy+p3.yxx)*p3.zyx);}vec2 posToUv(vec3 pos){vec2 uv=pos.xy/u_screenBounds.xy;uv=(uv+vec2(1.0))/2.0;uv.y=1.0-uv.y;return uv;}void main(){vec4 positionLife=texture2D(u_prevPosTex,v_uv);vec4 velInfo=texture2D(u_prevVelTex,v_uv);positionLife.w-=u_deltaTime*u_simDieSpeed*0.01;if(positionLife.w<0.0){vec3 h=hash33(vec3(v_uv,u_time));float modeCut=step(u_logoCutPercent,h.x);velInfo.w=(modeCut*h.y*2.0+1.0)*u_mode;}velInfo.xyz*=0.975;if(velInfo.w*u_mode>1.0){vec3 originPos=texture2D(u_logoPosTex,v_uv).xyz;vec3 attrV=originPos-positionLife.xyz;float dist2=dot(attrV,attrV);if(dist2>0.0001){attrV/=sqrt(dist2);}else{attrV=vec3(1.0,0.0,0.0);}velInfo.xyz+=attrV*u_attractForce*velInfo.w*u_deltaTime;}vec3 windVel=u_windForce*u_deltaTime*u_windStrMul;windVel/=1.0+velInfo.w*u_mode;velInfo.xyz+=windVel;vec2 posUv=posToUv(positionLife.xyz);vec3 mousePaintVel=(texture2D(u_mousePaintTex,posUv).xyz-0.5+0.001)*2.0;mousePaintVel.z=0.0;vec3 mouseFinalVel=mousePaintVel*0.8*u_mouseMoveIntensity*u_mouseStrength;mouseFinalVel*=1.0+velInfo.w*0.5*u_mode;velInfo.xyz+=mouseFinalVel;gl_FragColor=velInfo;}`;
function packTexture(et, $) {
  let tt = et.attributes;
  let nt = {
    x: 0,
    y: 0
  };
  let rt = $.pack.length;
  let it = Object.values(tt)[0].count;
  let ot = $.vertexCount || it;
  let st = Math.ceil(Math.sqrt(ot));
  nt.x = st * rt;
  nt.y = st;
  let lt = nt.x * nt.y * 4;
  let ut = new Float32Array(lt);
  let ct = 0;
  let ht = {};
  $.pack.flat().forEach(gt => {
    if (ht[gt] === undefined) {
      ht[gt] = 0;
    }
  });
  for (let gt = 0; gt < it; gt++) {
    for (let vt = 0; vt < rt; vt++) {
      let yt = $.pack[vt];
      for (let wt = 0; wt < 4; wt++) {
        let mt = yt[wt];
        if (mt !== undefined) {
          let _t = ht[mt];
          ut[ct] = tt[mt].array[_t];
          ht[mt] += 1;
        } else {
          ut[ct] = 0;
        }
        ct++;
      }
    }
  }
  while (ct < lt) {
    ut[ct] = 0;
    ct++;
  }
  return {
    texture: fboHelper.createDataTexture(ut, nt.x, nt.y, true, true, true),
    textureSize: nt
  };
}
const TEX_SIZE = 128;
const PARTICLE_COUNT = TEX_SIZE * TEX_SIZE;
const GUI = settings.LOOK_DEV_MODE;
const SPAWN = {
  bounds: {
    x: 4,
    y: 2.4,
    z: 0.64
  },
  offset: {
    x: -3,
    y: -0.5,
    z: 0
  }
};
const KILL = {
  bounds: {
    x: 7,
    y: 5,
    z: 2
  }
};
class Particles {
  constructor() {
    Et(this, 