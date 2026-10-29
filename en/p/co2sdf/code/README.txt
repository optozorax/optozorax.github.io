Complex distance fields: the reference code
============================================

The code behind the numbers and pictures of the post "Complex distance fields: SDF worlds one
dimension up" (Part V of the post explains everything below in more detail).
Python 3 with numpy and pillow (matplotlib for a few checks); a C++17 compiler with OpenMP for
the 3D tracer.

Kernel and scenes
-----------------
co2sdf.py       the kernel: Jacobian by central differences, reciprocal frame, Gauss-Newton
                projection, tangent projection, second-order geodesic step, and the operators
                union, fold_union (= smooth_union), neck_union. At the end, the post's library
                under the post's names: ball, extrude, smin, capsule, world, smooth_union,
                extrude_world, turn, roots.
scenes.py       every 2D-in-R^4 scene of the post, written with that library. Scenes with hard
                unions, necks, mirror cuts or several charts define jac(), which takes the
                gradients with the branch or chart frozen at the point.
scenes5d.py, render3d.py, render5d.py   the first 3D-in-R^5 scenes in numpy.
render6d.py     the codimension-3 inversion neck in R^6 and a k x k Gram-solve kernel.
pga.py          a tiny dense geometric algebra R(n,0,1) for the PGA checks.

Rendering
---------
python3 render.py neck                                  quick preview scan
python3 render_hq.py scan <scene> 640 10 8192 10000     page-quality scan (WORKERS=<n>, CURV=1)
g++ -O3 -march=native -fopenmp -std=c++17 trace3d.cpp -o trace3d
FOG=0.006 ./trace3d <scene> 640 360 10 10000 out.rgbf   3D view, raw float32 RGB
./trace3d <scene> eval < points.txt                     the field at points, 5 numbers per line
NOSELF=1                                                hide the camera ball in the 3D views

Checks (each prints its result)
-------------------------------
checks_composable.py    the composable constructions: where projected points land, independent
                        gradients, flatness and curvature claims, orientation through the handles
test_kernel.py          projection, geodesics, neck regularity, cone angles
test_integrators*.py    convergence order of the geodesic step (first order vs the chord update)
test_klein.py           frame transport through the ordinary and the mirror handle
test_pga.py             the PGA form of the push-back step
cmp_cpp2.py             the C++ tracer against the numpy tracer

Figures
-------
fig3d.py is a small numpy ray tracer of the explanatory 3D figures; it also writes their SVG
labels. fig_planes.py, fig_push.py, fig_meaning.py, fig_circle.py and fig_mirror.py make the five
figures, preview.py draws the labels onto the PNG, mirror_fig.py draws the older frame-transport
diagram of the mirror handle.

The live widget is in ../demo/ (open index.html), and the ball-shaped wormhole into itself,
built by codex (gpt-6-astra) with its checks, is in ../mono_ball/.
