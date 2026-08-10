---
canonical: true
quality: 0.98
source_kind: threejs_production_reference
---
# Three.js Performance and Rendering Reference

This reference covers measured frame budgets, draw calls, instancing, materials, lighting, responsive drawing buffers, render-loop strategy, interaction, and post-processing for production Three.js scenes.

## 4. Frame-budget and device strategy

Set budgets from the actual product and target hardware, then measure. Useful starting targets—not universal guarantees—are:

- 60 FPS target: under 16.7 ms total frame time; reserve headroom for layout, input, and browser work.
- 30 FPS fallback: under 33.3 ms on low-power devices.
- Draw calls: aim below roughly 100 on mobile and 200 on desktop for a product viewer; profile before accepting higher counts.
- Visible triangles: begin around 100k–250k mobile and 300k–750k desktop, then validate material complexity, overdraw, and fill rate.
- Pixel ratio: usually cap around 1.0–1.25 mobile and 1.5 desktop. Increase only when measured headroom exists.
- Shadow maps: start at 1024; use 2048 only when the visible quality improvement justifies memory and fill cost.
- Post-processing: keep the minimum pass count and render-target resolution needed for the visual goal.

Do not optimize only triangle count. A scene with few triangles can still be slow because of transparent overdraw, many materials, shader compilation, large render targets, excessive lights, shadows, or high DPR.

Use adaptive quality tiers when the experience must span phones and desktops. Quality tiers may alter DPR cap, antialiasing, shadow resolution, post-processing, LOD, particle count, reflection quality, and animation complexity. Make changes at stable boundaries; do not oscillate quality every frame.

## 5. Draw calls, instancing, merging, and LOD

- Reuse one geometry and material for repeated shapes.
- Use `InstancedMesh` for repeated objects that share geometry/material but require different transforms or instance colors.
- Mark instance matrices/colors dirty only after changes: `instanceMatrix.needsUpdate = true`.
- Merge static `BufferGeometry` with `BufferGeometryUtils.mergeGeometries` when objects share a compatible material and do not require independent culling or interaction.
- Avoid thousands of scene graph nodes. Node traversal and world-matrix updates cost CPU time even when geometry is simple.
- Use `LOD` or application-specific distance tiers for expensive models. Ensure transitions are acceptable and test camera movement.
- Preserve frustum culling. A single huge merged mesh may keep an entire region visible when only one small part intersects the camera.
- For repeated animated characters, ordinary `InstancedMesh` is insufficient for independent skeletal animation without a specialized approach. Do not pretend instancing solves every repeated model.

## 6. Materials, lights, shadows, and transparency

- Prefer `MeshStandardMaterial` or `MeshPhysicalMaterial` only where PBR features are visible. Physical transmission, clearcoat, sheen, and multiple lights increase shader cost.
- Reuse material instances when uniform values are identical. Different materials usually create separate draw calls even for identical geometry.
- Avoid changing material defines every frame. Changes such as toggling maps, transparency, skinning, or vertex colors can trigger shader variants and compilation.
- Warm critical shader variants during loading if first-interaction hitching is observable. Do not compile every hypothetical variant.
- Keep transparent meshes limited and sorted behavior understood. Transparency disables many early-depth advantages and can create visual ordering artifacts.
- Prefer alpha test or alpha hash for cutout foliage when appropriate rather than fully blended transparency.
- Let as few lights as possible cast shadows. Tighten directional-light shadow camera bounds around the subject, set sensible near/far values, and disable casting/receiving on objects that do not need it.
- Consider baked lighting, ambient occlusion, lightmaps, contact shadows, or blob shadows for mostly static scenes.

## 7. Responsive canvas and drawing-buffer control

CSS owns display size; Three.js owns drawing-buffer size. Never call `renderer.setSize` unconditionally every frame.

```js
function resizeRendererToDisplaySize(renderer, camera, dprCap = 1.5) {
  const canvas = renderer.domElement;
  const dpr = Math.min(window.devicePixelRatio || 1, dprCap);
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width === width && canvas.height === height) return false;

  renderer.setSize(width, height, false);
  camera.aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
  camera.updateProjectionMatrix();
  return true;
}
```

- Pass `false` to keep CSS responsible for layout.
- Guard zero-height containers during route transitions and hidden tabs.
- Resize post-processing composers and render targets with the drawing buffer.
- Do not blindly call `renderer.setPixelRatio(window.devicePixelRatio)` for a heavy scene; a DPR of 3 renders nine times as many pixels as DPR 1.
- Use `ResizeObserver` for embedded canvases whose container changes independently of the window.

## 8. Render loop and on-demand rendering

For continuously animated scenes:

- Use one RAF or `renderer.setAnimationLoop`, never both.
- Clamp large deltas after tab suspension.
- Keep simulation updates separate from rendering.
- Reuse temporary math objects.
- Pause expensive work when the document is hidden or the canvas is offscreen.

For model viewers and editors, render on demand:

```js
let renderRequested = false;
function requestRender() {
  if (renderRequested) return;
  renderRequested = true;
  requestAnimationFrame(() => {
    renderRequested = false;
    resizeRendererToDisplaySize(renderer, camera);
    controls.update();
    renderer.render(scene, camera);
  });
}

controls.addEventListener('change', requestRender);
window.addEventListener('resize', requestRender);
```

Call `requestRender` when assets load, controls change, selection changes, animation advances, UI changes material values, or the container resizes. If damping is enabled, continue rendering until controls settle.

## 9. Interaction, raycasting, and controls

- Convert pointer coordinates from the canvas bounding rectangle, not the whole window.
- Raycast on pointer movement only when hover feedback is needed. Throttle expensive picking and restrict targets by layers or an explicit pickable list.
- Avoid allocating a new array and vectors for every pointer event; reuse them.
- For very dense static geometry, consider a verified BVH acceleration library, but do not add it unless profiling shows raycasting is the bottleneck.
- Give canvas interactions keyboard-accessible HTML equivalents when they represent essential actions.
- Configure OrbitControls limits intentionally: min/max distance, polar angle, target, zoom, pan, damping, and touch behavior.
- Prevent controls from competing with page scroll or UI overlays. Pointer capture and `touch-action` must match the intended interaction.

## 10. Post-processing and visual polish

- Add post-processing after the base scene meets its frame budget.
- Every full-screen pass reads/writes many pixels. Bloom, SSAO, depth of field, motion blur, outlines, and antialiasing passes should be justified and measured.
- Render expensive effects at reduced resolution when acceptable.
- Use selective bloom or masks rather than making the entire scene glow.
- Keep tone mapping, exposure, environment, material values, and color management consistent before compensating with effects.
- Test transparency, UI overlays, screenshots, and resizing with the composer enabled.
