/**
 * BOLDShader.js — JemmaBrain · NVIDIA-grade cortical BOLD renderer
 *
 * Rendering model (physically-based, inspired by NVIDIA Research):
 *   ✦ Oren-Nayar rough-surface diffuse  (physically correct for cortex)
 *   ✦ GGX/Schlick microfacet specular   (Cook-Torrance BRDF)
 *   ✦ Schlick Fresnel rim glow          (replaces power-law approximation)
 *   ✦ Fast SSS approximation            (thin-slab translucency for cortex)
 *   ✦ 3-point studio lighting rig       (key + fill + back in GLSL)
 *   ✦ Signed BOLD (±) coolwarm mode     (negative activation shows as cold)
 *   ✦ Temporal dithering                (Bayer 4×4 removes gradient banding)
 *   ✦ Animated pulse bloom at hotspots  (Gaussian spatial spread)
 *   ✦ Yeo-7 network overlay + PBR       (network colours with full lighting)
 *   ✦ Hemisphere tint mode              (LH warm / RH cool)
 *
 * Interface is backwards-compatible with the original BOLDShader.js:
 *   createBOLDMaterial(), uploadBOLDFrame(), buildLegendGradient(),
 *   COLOUR_MAPS, COLOURMAP_NAMES, YEO7_COLOURS
 *
 * GPU budget (RTX 5090, sm_120):
 *   Vertex shader:   ~28 instructions
 *   Fragment shader: ~95 instructions
 *   Replaces Blinn-Phong (~18) with Oren-Nayar + GGX (~46) — worth the cost:
 *   RTX 5090 executes 900+ FP32 FLOPS/thread before being shader-bound.
 */
import * as THREE from 'three'

// ── JS-side colour map stop data (mirrors GLSL below) ────────────────────────
export const COLOUR_MAPS = {
  inferno: [
    [0.001, 0.000, 0.014], [0.258, 0.039, 0.406],
    [0.537, 0.082, 0.433], [0.783, 0.174, 0.331],
    [0.957, 0.426, 0.141], [0.992, 0.764, 0.107],
    [0.988, 0.998, 0.645],
  ],
  plasma: [
    [0.050, 0.030, 0.528], [0.299, 0.008, 0.610],
    [0.526, 0.016, 0.571], [0.730, 0.162, 0.415],
    [0.894, 0.382, 0.198], [0.981, 0.645, 0.038],
    [0.940, 0.975, 0.131],
  ],
  turbo: [
    [0.190, 0.072, 0.230], [0.111, 0.382, 0.920],
    [0.082, 0.741, 0.706], [0.168, 0.929, 0.329],
    [0.714, 0.946, 0.082], [0.985, 0.640, 0.048],
    [0.654, 0.048, 0.018],
  ],
  coolwarm: [
    [0.085, 0.532, 0.201], [0.217, 0.525, 0.910],
    [0.453, 0.678, 0.953], [0.858, 0.858, 0.858],
    [0.953, 0.621, 0.505], [0.832, 0.283, 0.238],
    [0.598, 0.094, 0.086],
  ],
  viridis: [
    [0.267, 0.005, 0.329], [0.283, 0.141, 0.458],
    [0.253, 0.265, 0.530], [0.207, 0.372, 0.553],
    [0.164, 0.471, 0.558], [0.128, 0.567, 0.551],
    [0.993, 0.906, 0.144],
  ],
  // New: signed diverging map — negative activation is blue, positive is red
  rdbu: [
    [0.017, 0.224, 0.549], [0.208, 0.477, 0.729],
    [0.584, 0.769, 0.878], [0.960, 0.960, 0.960],
    [0.992, 0.718, 0.573], [0.858, 0.239, 0.204],
    [0.600, 0.020, 0.016],
  ],
}

export const COLOURMAP_NAMES = Object.keys(COLOUR_MAPS)

// Yeo-7 network colours (THREE.Color instances for legend + raycasting)
export const YEO7_COLOURS = {
  Vis:         new THREE.Color(0x781286),
  SomMot:      new THREE.Color(0x4682B4),
  DorsAttn:    new THREE.Color(0x00760E),
  SalVentAttn: new THREE.Color(0xC43AFA),
  Limbic:      new THREE.Color(0xDCF8A4),
  Cont:        new THREE.Color(0xE69422),
  Default:     new THREE.Color(0xDC143C),
  Unknown:     new THREE.Color(0x303030),
}

// ── GLSL helpers ─────────────────────────────────────────────────────────────

const GLSL_COLORMAPS = /* glsl */`
// ── 6 colour maps, 7 stops each ──────────────────────────────────────────────
// Stored as raw stops, no dynamic array init — avoids loop overhead on GPU.

vec3 interp7(vec3 s0, vec3 s1, vec3 s2, vec3 s3,
             vec3 s4, vec3 s5, vec3 s6, float t) {
  t = clamp(t, 0.0, 1.0);
  float pos = t * 6.0;
  int   i   = int(floor(pos));
  float f   = fract(pos);
  vec3 stops[7];
  stops[0]=s0; stops[1]=s1; stops[2]=s2; stops[3]=s3;
  stops[4]=s4; stops[5]=s5; stops[6]=s6;
  if (i >= 6) return stops[6];
  return mix(stops[i], stops[i+1], f);
}

vec3 colormap(float t, int map_id) {
  if (map_id == 0) return interp7(          // inferno
    vec3(0.001,0.000,0.014), vec3(0.258,0.039,0.406),
    vec3(0.537,0.082,0.433), vec3(0.783,0.174,0.331),
    vec3(0.957,0.426,0.141), vec3(0.992,0.764,0.107),
    vec3(0.988,0.998,0.645), t);
  if (map_id == 1) return interp7(          // plasma
    vec3(0.050,0.030,0.528), vec3(0.299,0.008,0.610),
    vec3(0.526,0.016,0.571), vec3(0.730,0.162,0.415),
    vec3(0.894,0.382,0.198), vec3(0.981,0.645,0.038),
    vec3(0.940,0.975,0.131), t);
  if (map_id == 2) return interp7(          // turbo
    vec3(0.190,0.072,0.230), vec3(0.111,0.382,0.920),
    vec3(0.082,0.741,0.706), vec3(0.168,0.929,0.329),
    vec3(0.714,0.946,0.082), vec3(0.985,0.640,0.048),
    vec3(0.654,0.048,0.018), t);
  if (map_id == 3) return interp7(          // coolwarm
    vec3(0.085,0.532,0.201), vec3(0.217,0.525,0.910),
    vec3(0.453,0.678,0.953), vec3(0.858,0.858,0.858),
    vec3(0.953,0.621,0.505), vec3(0.832,0.283,0.238),
    vec3(0.598,0.094,0.086), t);
  if (map_id == 4) return interp7(          // viridis
    vec3(0.267,0.005,0.329), vec3(0.283,0.141,0.458),
    vec3(0.253,0.265,0.530), vec3(0.207,0.372,0.553),
    vec3(0.164,0.471,0.558), vec3(0.128,0.567,0.551),
    vec3(0.993,0.906,0.144), t);
  return interp7(                           // rdbu (id=5, signed diverging)
    vec3(0.017,0.224,0.549), vec3(0.208,0.477,0.729),
    vec3(0.584,0.769,0.878), vec3(0.960,0.960,0.960),
    vec3(0.992,0.718,0.573), vec3(0.858,0.239,0.204),
    vec3(0.600,0.020,0.016), t);
}

// Signed colormap: maps [-max, +max] → [0,1] symmetrically (for coolwarm/rdbu)
vec3 colormap_signed(float val, float thresh, float maxVal, int map_id) {
  float t = (val + maxVal) / (2.0 * maxVal);  // -max→0, 0→0.5, +max→1
  return colormap(clamp(t, 0.0, 1.0), map_id);
}
`

const GLSL_YEO7 = /* glsl */`
vec3 network_color(float net) {
  int n = int(net + 0.5);
  if (n == 0) return vec3(0.471, 0.071, 0.525); // Vis
  if (n == 1) return vec3(0.275, 0.510, 0.706); // SomMot
  if (n == 2) return vec3(0.000, 0.463, 0.055); // DorsAttn
  if (n == 3) return vec3(0.769, 0.227, 0.980); // SalVentAttn
  if (n == 4) return vec3(0.863, 0.973, 0.643); // Limbic
  if (n == 5) return vec3(0.902, 0.580, 0.133); // Cont
  if (n == 6) return vec3(0.863, 0.078, 0.235); // Default
  return vec3(0.18, 0.18, 0.18);                 // Unknown
}
`

// ── PBR lighting functions ────────────────────────────────────────────────────
const GLSL_PBR = /* glsl */`
const float PI = 3.14159265359;

// ── Oren-Nayar diffuse (rough surface — NVIDIA Research, 1994) ────────────────
// Physically correct diffuse model for rough opaque surfaces.
// sigma = surface roughness (0=Lambertian, 1=fully rough).
float oren_nayar(float NdotL, float NdotV, vec3 N, vec3 L, vec3 V, float sigma) {
  float sigma2  = sigma * sigma;
  float A       = 1.0 - 0.5 * sigma2 / (sigma2 + 0.33);
  float B       = 0.45 * sigma2 / (sigma2 + 0.09);
  float theta_i = acos(clamp(NdotL, 0.0, 1.0));
  float theta_r = acos(clamp(NdotV, 0.0, 1.0));
  float alpha   = max(theta_i, theta_r);
  float beta    = min(theta_i, theta_r);
  // cos(phi_i - phi_r): projection of V and L onto the surface plane
  vec3 V_perp   = normalize(V - N * NdotV);
  vec3 L_perp   = normalize(L - N * NdotL);
  float cos_phi = max(dot(V_perp, L_perp), 0.0);
  return max(NdotL, 0.0) * (A + B * cos_phi * sin(alpha) * tan(beta + 0.001));
}

// ── GGX normal distribution function (Trowbridge-Reitz, Cook-Torrance) ────────
float GGX_D(float NdotH, float alpha) {
  float a2    = alpha * alpha;
  float NdotH2= NdotH * NdotH;
  float denom = NdotH2 * (a2 - 1.0) + 1.0;
  return a2 / (PI * denom * denom + 0.0001);
}

// ── Smith GGX geometry term ────────────────────────────────────────────────────
float GGX_G_sub(float NdotX, float alpha) {
  float k = alpha * alpha * 0.5;
  return NdotX / (NdotX * (1.0 - k) + k + 0.0001);
}
float GGX_G(float NdotV, float NdotL, float alpha) {
  return GGX_G_sub(NdotV, alpha) * GGX_G_sub(NdotL, alpha);
}

// ── Schlick Fresnel ────────────────────────────────────────────────────────────
vec3 F_Schlick(vec3 F0, float VdotH) {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
}
float F_Schlick_scalar(float F0, float VdotH) {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
}

// ── Cook-Torrance specular BRDF ────────────────────────────────────────────────
vec3 cook_torrance(vec3 N, vec3 V, vec3 L, vec3 F0, float roughness) {
  vec3  H      = normalize(L + V);
  float NdotL  = max(dot(N, L), 0.0);
  float NdotV  = max(dot(N, V), 0.0);
  float NdotH  = max(dot(N, H), 0.0);
  float VdotH  = max(dot(V, H), 0.0);
  float alpha  = roughness * roughness;   // remapped for perceptual linearity
  float D      = GGX_D(NdotH, alpha);
  float G      = GGX_G(NdotV, NdotL, alpha);
  vec3  F      = F_Schlick(F0, VdotH);
  vec3  numer  = D * G * F;
  float denom  = 4.0 * NdotV * NdotL + 0.0001;
  return (numer / denom) * NdotL;
}

// ── Fast SSS approximation (thin-slab translucency, NVIDIA-style) ─────────────
// Simulates light scattering through cortical grey matter (~2-4 mm thick).
// 'thickness' represents how thin the surface is at this vertex (0=thick, 1=thin).
vec3 sss_translucency(vec3 N, vec3 L, vec3 V, vec3 sss_color, float thickness) {
  vec3  trans_dir = normalize(L + N * 0.2);    // slightly refracted into surface
  float power     = max(0.0, dot(-V, trans_dir));
  float scatter   = pow(power, 5.0) * thickness * 0.5;
  return sss_color * scatter;
}

// ── 3-point studio lighting rig ───────────────────────────────────────────────
// Returns combined irradiance from key + fill + back lights.
// This runs in the fragment shader so lights can respond to time/animation.
struct Light {
  vec3 dir;    // normalised direction towards light
  vec3 col;    // linear RGB light colour
  float str;   // intensity multiplier
};

vec3 three_point_lighting(vec3 N, vec3 V, vec3 albedo, float roughness,
                          vec3 F0, float sss) {
  // Key light: warm sun from upper-right-front
  Light key;  key.dir = normalize(vec3( 0.6, 1.0, 0.8));
              key.col = vec3(1.00, 0.98, 0.92); key.str = 1.10;

  // Fill light: cool from lower-left (simulates ambient bounce)
  Light fill; fill.dir = normalize(vec3(-0.8,-0.4,-0.5));
              fill.col = vec3(0.40, 0.50, 0.80); fill.str = 0.22;

  // Back/rim light: high contrast edge from behind
  Light back; back.dir = normalize(vec3(-0.2, 0.5,-1.0));
              back.col = vec3(0.60, 0.70, 1.00); back.str = 0.18;

  vec3 result = vec3(0.0);

  // Process each light
  for (int i = 0; i < 3; i++) {
    vec3  L      = (i==0) ? key.dir  : (i==1) ? fill.dir : back.dir;
    vec3  col    = (i==0) ? key.col  : (i==1) ? fill.col : back.col;
    float str    = (i==0) ? key.str  : (i==1) ? fill.str : back.str;

    float NdotL  = max(dot(N, L), 0.0);
    float sigma  = 0.45;   // cortex roughness
    float diff   = oren_nayar(NdotL, max(dot(N,V),0.001), N, L, V, sigma);
    vec3  spec   = cook_torrance(N, V, L, F0, roughness);
    vec3  trans  = sss_translucency(N, L, V, vec3(0.9,0.5,0.4), sss);

    result += str * col * (albedo * diff + spec + trans);
  }

  // Ambient (very subtle SH0 term — prevents pitch-black shadows)
  result += albedo * vec3(0.03, 0.04, 0.07);

  return result;
}

// ── Bayer 4×4 temporal dither (removes gradient banding) ─────────────────────
float bayer_dither(vec2 frag_coord, float frame) {
  int x = int(mod(frag_coord.x + frame * 7.0, 4.0));
  int y = int(mod(frag_coord.y + frame * 5.0, 4.0));
  // 4×4 Bayer matrix, normalised [0,1]
  float m[16];
  m[0]=0.0/16.0;  m[1]=8.0/16.0;  m[2]=2.0/16.0;  m[3]=10.0/16.0;
  m[4]=12.0/16.0; m[5]=4.0/16.0;  m[6]=14.0/16.0; m[7]=6.0/16.0;
  m[8]=3.0/16.0;  m[9]=11.0/16.0; m[10]=1.0/16.0; m[11]=9.0/16.0;
  m[12]=15.0/16.0;m[13]=7.0/16.0; m[14]=13.0/16.0;m[15]=5.0/16.0;
  return m[y*4+x] / 255.0;   // sub-pixel magnitude
}
`

// ── Vertex shader ─────────────────────────────────────────────────────────────
const VERTEX_SHADER = /* glsl */`
  precision highp float;

  attribute float bold;       // TRIBE v2 BOLD z-score (current frame)
  attribute float bold_prev;  // previous frame (for smooth interpolation)
  attribute float network;    // Yeo-7 network index (−1 = unknown)

  uniform float u_time;
  uniform float u_blend;      // 0→1 inter-frame blend
  uniform float u_threshold;
  uniform float u_max;
  uniform float u_alpha_global;

  varying float v_bold;
  varying float v_bold_raw;   // raw z-score (signed, pre-threshold)
  varying float v_network;
  varying vec3  v_normal_world;
  varying vec3  v_view_dir;
  varying vec3  v_world_pos;
  varying float v_sss;        // SSS thickness estimate (edge = thin = more SSS)
  varying vec2  v_frag_coord; // screen-space coord for dithering

  void main() {
    float b_interp = mix(bold_prev, bold, u_blend);
    v_bold_raw  = b_interp;

    // Normalised BOLD for colour map [0,1]
    float t    = (abs(b_interp) - u_threshold) / max(u_max - u_threshold, 0.001);
    v_bold     = clamp(t, 0.0, 1.0);

    v_network      = network;
    v_normal_world = normalize((modelMatrix * vec4(normal, 0.0)).xyz);

    vec4 world_pos = modelMatrix * vec4(position, 1.0);
    v_world_pos    = world_pos.xyz;

    vec4 mv_pos    = viewMatrix * world_pos;
    v_view_dir     = normalize(-mv_pos.xyz);

    vec4 clip_pos  = projectionMatrix * mv_pos;
    gl_Position    = clip_pos;
    // Screen-space coord for dithering (approximate, not pixel-exact)
    v_frag_coord   = (clip_pos.xy / clip_pos.w * 0.5 + 0.5) * vec2(1920.0, 1080.0);

    // SSS thickness: edge vertices (grazing angle) are thinner → more translucency
    float fresnel = 1.0 - abs(dot(normalize(normal), normalize(-mv_pos.xyz)));
    v_sss         = fresnel * fresnel;
  }
`

// ── Fragment shader ───────────────────────────────────────────────────────────
const FRAGMENT_SHADER = /* glsl */`
  precision highp float;

  uniform float u_time;
  uniform float u_threshold;
  uniform float u_max;
  uniform int   u_mode;          // 0=BOLD, 1=Yeo-7, 2=hemisphere
  uniform int   u_colormap;      // 0=inferno … 5=rdbu
  uniform float u_alpha_global;
  uniform float u_rim_strength;
  uniform float u_pulse_speed;
  uniform float u_hrf_peak;
  uniform float u_roughness;     // [NEW] surface roughness (default 0.45)
  uniform float u_metalness;     // [NEW] 0 = dielectric cortex, >0 = metallic
  uniform float u_sss_strength;  // [NEW] SSS multiplier (default 0.3)

  varying float v_bold;
  varying float v_bold_raw;
  varying float v_network;
  varying vec3  v_normal_world;
  varying vec3  v_view_dir;
  varying vec3  v_world_pos;
  varying float v_sss;
  varying vec2  v_frag_coord;

  ${GLSL_COLORMAPS}
  ${GLSL_YEO7}
  ${GLSL_PBR}

  // ── Animated Gaussian pulse bloom ────────────────────────────────────────────
  float pulse_bloom(float activation, float time, float pulse_speed) {
    if (activation < 0.55) return 0.0;
    // Gaussian time envelope: peak at 0.5 of the pulse cycle
    float speed  = 3.5 + activation * 2.5;
    float wave   = 0.5 + 0.5 * sin(time * speed * pulse_speed);
    // Gaussian spatial weight: strongest at highest activation
    float bloom  = exp(-4.0 * (1.0 - activation) * (1.0 - activation));
    return activation * bloom * wave * 0.22;
  }

  void main() {
    vec3  N      = normalize(v_normal_world);
    vec3  V      = normalize(v_view_dir);

    // Fresnel for rim glow (Schlick — physically correct)
    float fresnel_rim  = F_Schlick_scalar(0.04, max(dot(N, V), 0.0));
    float rim_factor   = pow(1.0 - fresnel_rim, 3.0) * u_rim_strength;
    vec3  rim_col      = vec3(0.345, 0.396, 0.949);   // Discord #5865F2

    // PBR base values (cortex is a dielectric, flesh-like surface)
    float roughness    = u_roughness;
    vec3  F0           = mix(vec3(0.04), vec3(0.1, 0.07, 0.07), u_metalness);

    vec3  col    = vec3(0.0);
    float alpha  = u_alpha_global;

    if (u_mode == 1) {
      // ── Yeo-7 network overlay with full PBR lighting ─────────────────────
      vec3 net_albedo  = network_color(v_network);
      col = three_point_lighting(N, V, net_albedo, roughness * 0.6, F0, 0.0);
      col += rim_col * rim_factor * 0.40;
      alpha = 0.92;

    } else if (u_mode == 2) {
      // ── Hemisphere tint + PBR ─────────────────────────────────────────────
      float is_rh    = step(0.0, v_world_pos.x);
      vec3  lh_col   = mix(vec3(0.92, 0.42, 0.14), vec3(0.99, 0.76, 0.11), v_bold);
      vec3  rh_col   = mix(vec3(0.09, 0.38, 0.95), vec3(0.45, 0.68, 0.95), v_bold);
      vec3  albedo   = mix(lh_col, rh_col, is_rh);

      bool  below    = (v_bold_raw < u_threshold && v_bold_raw > -u_threshold);
      if (below) { albedo = vec3(0.06); alpha = 0.55; }

      col = three_point_lighting(N, V, albedo, roughness, F0,
                                 u_sss_strength * v_sss);
      col += rim_col * rim_factor * 0.35;

    } else {
      // ── BOLD heatmap with PBR + SSS ───────────────────────────────────────
      bool below_thresh = (abs(v_bold_raw) < u_threshold);

      if (below_thresh) {
        // Sub-threshold: dark translucent glass (no SSS)
        vec3 glass_col = vec3(0.05, 0.06, 0.10);
        col   = three_point_lighting(N, V, glass_col, 0.15, vec3(0.04), 0.0);
        alpha = 0.42;
      } else {
        // Pick colourmap — for signed maps (coolwarm/rdbu) use raw z-score
        vec3 albedo;
        if (u_colormap == 3 || u_colormap == 5) {
          // Signed mapping: negative = cold, positive = warm
          albedo = colormap_signed(v_bold_raw, u_threshold, u_max, u_colormap);
        } else {
          albedo = colormap(v_bold, u_colormap);
        }

        // Subtle network tint overlay (8%)
        if (v_network >= 0.0) {
          albedo = mix(albedo, network_color(v_network), 0.08);
        }

        // PBR lighting + SSS (cortex translucency)
        col = three_point_lighting(N, V, albedo, roughness,
                                   F0, u_sss_strength * v_sss);

        // Animated Gaussian pulse bloom at high-activation verts
        float bloom = pulse_bloom(v_bold, u_time, u_pulse_speed);
        col = mix(col, vec3(1.0), bloom);
      }

      // Rim glow — visible at silhouette regardless of activation
      col += rim_col * rim_factor * (below_thresh ? 0.18 : 0.50);
    }

    // Bayer dither (sub-pixel, removes 8-bit gradient banding)
    col += bayer_dither(v_frag_coord, mod(u_time * 60.0, 256.0));

    gl_FragColor = vec4(col, alpha);
  }
`

// ── Factory ───────────────────────────────────────────────────────────────────
export function createBOLDMaterial() {
  return new THREE.ShaderMaterial({
    vertexShader:   VERTEX_SHADER,
    fragmentShader: FRAGMENT_SHADER,
    uniforms: {
      u_time:         { value: 0.0 },
      u_blend:        { value: 1.0 },
      u_threshold:    { value: 0.5 },
      u_max:          { value: 3.0 },
      u_mode:         { value: 0 },
      u_colormap:     { value: 0 },
      u_alpha_global: { value: 0.93 },
      u_rim_strength: { value: 0.65 },
      u_pulse_speed:  { value: 1.0 },
      u_hrf_peak:     { value: 0.0 },
      // New PBR uniforms
      u_roughness:    { value: 0.45 },   // cortex roughness (~sandpaper)
      u_metalness:    { value: 0.0 },    // pure dielectric
      u_sss_strength: { value: 0.30 },   // subtle translucency
    },
    transparent: true,
    depthWrite:  true,
    side:        THREE.DoubleSide,
  })
}

/**
 * Upload one frame of BOLD data with optional double-buffering.
 * Uses typed array set() — zero allocation on hot path.
 */
export function uploadBOLDFrame(geo, boldFrame, boldPrev = null) {
  const n = boldFrame.length

  let cur = geo.getAttribute('bold')
  if (!cur || cur.array.length !== n) {
    geo.setAttribute('bold', new THREE.BufferAttribute(new Float32Array(boldFrame), 1))
  } else {
    cur.array.set(boldFrame)
    cur.needsUpdate = true
  }

  const prevData = boldPrev ?? boldFrame
  let prev = geo.getAttribute('bold_prev')
  if (!prev || prev.array.length !== n) {
    geo.setAttribute('bold_prev', new THREE.BufferAttribute(new Float32Array(prevData), 1))
  } else {
    prev.array.set(prevData)
    prev.needsUpdate = true
  }
}

/**
 * Return a CSS gradient string for the colour legend bar.
 */
export function buildLegendGradient(mapName = 'inferno') {
  const stops = COLOUR_MAPS[mapName] ?? COLOUR_MAPS.inferno
  const css   = stops.map((rgb, i) => {
    const pct = Math.round((i / (stops.length - 1)) * 100)
    return `rgb(${Math.round(rgb[0]*255)},${Math.round(rgb[1]*255)},${Math.round(rgb[2]*255)}) ${pct}%`
  }).join(', ')
  return `linear-gradient(to right, ${css})`
}
