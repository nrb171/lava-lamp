#version 300 es
precision highp float;
in vec2 vW;
in float vU;
out vec4 o;
void main() { o = vec4(vW * exp(-0.5 * vU * vU), 0.0, 0.0); }
