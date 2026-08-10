---
canonical: true
quality: 0.98
source_kind: threejs_production_reference
---
# Three.js Lifecycle and Verification Reference

This reference covers GPU resource ownership, cleanup, instrumentation, diagnosis, accessibility, framework lifecycle, failure patterns, and end-to-end delivery checks for production Three.js scenes.

## 11. Memory and cleanup

Three.js cannot automatically free GPU allocations while the page remains active. Dispose resources when replacing models, changing routes, or unmounting components.

```js
function disposeMaterial(material) {
  for (const value of Object.values(material)) {
    if (value?.isTexture) value.dispose();
  }
  material.dispose();
}

function disposeObject(root) {
  root.traverse((object) => {
    object.geometry?.dispose?.();
    if (Array.isArray(object.material)) {
      object.material.forEach(disposeMaterial);
    } else if (object.material) {
      disposeMaterial(object.material);
    }
  });
  root.removeFromParent();
}
```

Also dispose render targets, post-processing passes/composers where supported, controls, PMREM generators, KTX2/Draco workers where applicable, and custom GPU resources. Remove event listeners, observers, timers, and RAF callbacks. Avoid double-disposing shared assets; use explicit ownership or reference counting when multiple scenes share resources.

## 12. Instrumentation and diagnosis

During development expose or log:

```js
const info = renderer.info;
console.table({
  calls: info.render.calls,
  triangles: info.render.triangles,
  lines: info.render.lines,
  points: info.render.points,
  geometries: info.memory.geometries,
  textures: info.memory.textures,
});
```

Use browser Performance and Memory tools plus GPU-aware tools such as Spector.js when available. Diagnose in this order:

1. Confirm drawing-buffer dimensions and DPR.
2. Record frame time and determine CPU-bound versus GPU/fill-bound behavior.
3. Check draw calls, triangles, shader variants, transparent overdraw, lights, shadows, and post-processing passes.
4. Check asset download/decode/transcode time and first shader compilation stalls.
5. Check whether geometries/textures grow after repeated navigation or model replacement.
6. Change one variable, repeat the same camera path, and compare measurements.

Do not infer performance from a static screenshot or source complexity.

## 13. Accessibility, motion, and HTML integration

A canvas is not a complete interface. Keep headings, labels, descriptions, controls, prices, actions, loading progress, and errors in semantic HTML. Use the 3D view as progressive enhancement.

- Respect `prefers-reduced-motion`; reduce auto-rotation, camera fly-throughs, particles, and continuous background motion.
- Provide pause controls for nonessential animation.
- Keep focus indicators and keyboard controls visible.
- Ensure overlays remain readable over bright and dark scene regions.
- Do not trap wheel, touch, or keyboard input unnecessarily.
- Provide a fallback image or useful explanation when WebGL initialization or model loading fails.

## 14. Framework lifecycle notes

### React

Initialize in an effect tied to the canvas, return complete cleanup, and keep mutable Three.js objects in refs rather than React state. Do not recreate the renderer or scene for ordinary prop changes. In Strict Mode, tolerate mount-cleanup-mount.

### Vue and Svelte

Create the renderer after the canvas exists and dispose it in `onUnmounted`/`onDestroy`. Watch only values that must update the scene; do not rebuild everything for reactive changes.

### Plain JavaScript

Own initialization and teardown explicitly. If navigation is client-side, call teardown before replacing the route. Avoid hidden module globals that retain scenes after the canvas is gone.

### React Three Fiber

If the repository already uses R3F, follow its canvas, `useFrame`, disposal, loader cache, and invalidation conventions. Do not install R3F merely because the task mentions React and Three.js.

## 15. Failure patterns to reject

- Importing Three.js from a CDN inside a bundled app that already depends on `three`.
- Creating a renderer, scene, loaders, or materials during every component render.
- Starting multiple RAF loops after hot reload or route navigation.
- Rendering at unrestricted device pixel ratio.
- Adding many individual meshes where instancing or shared geometry is appropriate.
- Using dozens of dynamic lights or shadow-casting objects without measurement.
- Loading huge uncompressed textures and assuming small network size means small GPU size.
- Marking every material transparent to solve one alpha issue.
- Calling `renderer.setSize` every frame even when dimensions are unchanged.
- Forgetting camera aspect and projection updates on resize.
- Disposing shared textures while another object still uses them, or never disposing replaced assets.
- Building all controls inside the canvas when accessible HTML controls are simpler.
- Claiming 60 FPS without recording frame behavior on the target viewport/device class.

## 16. Delivery and verification checklist

Before declaring a Three.js task complete:

1. The requested scene and interaction work end to end.
2. Initial loading, progress, failure, empty, and fallback states are intentional.
3. Camera framing and controls work for the real asset dimensions.
4. Canvas resizing preserves aspect and uses a bounded DPR.
5. Color space, tone mapping, environment, material colors, and texture color spaces are correct.
6. Draw calls, triangles, textures, and frame time were observed on the representative scene.
7. Repeated objects share resources or use instancing where appropriate.
8. Shadow and post-processing costs are bounded and visually justified.
9. No loaders, geometries, materials, textures, render targets, controls, listeners, observers, or loops leak after teardown.
10. Narrow and wide layouts work; touch and pointer controls do not fight the page.
11. Reduced motion and a non-canvas fallback are available where needed.
12. The production build resolves all model, decoder, transcoder, HDR, and texture URLs.
13. The final result was exercised in a browser, not accepted from compilation alone.

A high-performing Three.js result is not the scene with the fewest features. It is the scene that preserves the intended visual effect and interaction while maintaining measured frame-time headroom, predictable memory ownership, responsive behavior, and graceful degradation on the actual target devices.

## 17. Authoritative references

- Three.js manual: https://threejs.org/manual/
- Fundamentals: https://threejs.org/manual/en/fundamentals.html
- Responsive rendering: https://threejs.org/manual/en/responsive.html
- Rendering on demand: https://threejs.org/manual/en/rendering-on-demand.html
- Optimizing many objects: https://threejs.org/manual/en/optimize-lots-of-objects.html
- Cleanup and disposal: https://threejs.org/manual/en/cleanup.html
- Three.js API documentation: https://threejs.org/docs/
