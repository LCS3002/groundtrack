# groundtrack

Turns oblique video, filmed from a fixed tripod or window, into accurate top-down movement
data: tracks in metres in **British National Grid (EPSG:27700)**. The output is ready for QGIS
(maps) and Houdini (particles and paths). Everything runs locally.

| tracks over the map (`topdown.png`) | smoothed flow field (`flow_field.png`) | animated (`topdown.mp4`) |
|---|---|---|
| ![topdown](docs/img/topdown.jpg) | ![flow field](docs/img/flow_field.jpg) | ![animation](docs/img/topdown_video.jpg) |

*Images from the synthetic demo (`groundtrack demo`): two people walking at known speeds, one
of them behind a pillar (white dotted = tracker prediction through the occlusion).*

```
video ──► 1 DETECT + TRACK ──► raw_tracks.csv          (YOLO26 + ByteTrack / BoT-SORT)
                │
GeoTIFF ─► 2 CALIBRATE (once per site) ──► homography.json   (6–8 clicked point pairs)
                │
          3 PROJECT TO GROUND   foot point → metres, vehicles shifted to their centre
          4 CLEAN + VECTORS     spikes, ID switches, occlusion stitching, Savitzky–Golay,
                │               vx vy speed heading, per-track metrics
          5 EXPORTS             points.csv · tracks.geojson · field_grid.csv · stats.json/csv
                │               vector_field.csv · houdini_import.py · houdini_field.py
          6 VISUALS             topdown.png/.mp4 · flow_field.png · density.png
                                speed_histogram.png · (debug.mp4 overlay)
```

---

## 1. Install

You need Python 3.10–3.12. Use a virtual environment.

### Windows / Linux + NVIDIA GPU

Install the CUDA build of PyTorch. RTX 50-series (Blackwell) cards need **CUDA 12.8 or
newer**, so `cu128` is a safe choice for any recent card:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
groundtrack device        # -> selected device: cuda:0, GPU: NVIDIA GeForce RTX ...
```

### Apple Silicon (M1–M4)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision          # the default macOS wheels include MPS
pip install -e ".[dev]"
groundtrack device                     # -> selected device: mps
```

### CPU only

`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`, then
`pip install -e ".[dev]"`. This works, but expect a few frames per second. Use `vid_stride: 2` and
`yolo26s.pt`.

`device: auto` in a site config picks **cuda → mps → cpu**. If you ask for `cuda` or `mps`
explicitly and it isn't available, the tool stops with an error rather than quietly running for
hours on the CPU. YOLO weights download into `models/` on first use.

### Try it without footage

```powershell
groundtrack demo --out demo_site
```

This builds a synthetic site: a virtual camera, a GeoTIFF, a video of two people walking at
known speeds, one of them passing behind a pillar. It then calibrates and runs the whole
pipeline. Look in `demo_site/runs/demo/demo/`.

---

## 2. Filming

* **Tripod, locked off.** The homography only holds while the camera doesn't move. Even a slight
  knock means you have to re-calibrate.
* **Stabilisation off** (iPhone: Settings → Camera → Record Video → Enhanced Stabilisation
  off; Android: turn off "video stabilisation"). Electronic stabilisation crops and shifts the
  frame over time, which breaks the calibration.
* **Use the 1× (main) lens.** Don't zoom, and avoid the 0.5× ultra-wide: its distortion bends
  straight kerbs (or use the lens calibration, §3.4). The phone must not switch lenses during a
  take. Lock focus and exposure (long-press on iPhone).
* **Fixed frame rate.** Turn off "Auto FPS" / "Auto Low Light FPS". Variable frame rate makes
  speeds wrong. 1080p30 or 4K30 are both fine.
* **Height helps.** The steeper the view, the more accurate the far field. A first-floor window
  beats street level by a long way.
* **Through a window:** put the lens right against the glass, and shade it with a dark cloth or
  hood. Turn room lights off. Reflections create ghost "people".
* Record the GeoTIFF area generously, and make sure the ground you film has **6–8 sharp,
  identifiable features** that are also visible on the map (§3.1).

---

## 3. Calibration walkthrough (once per site)

### 3.1 Make the georeferenced top-down image

The map must be a **north-up GeoTIFF in EPSG:27700**.

* **Digimap (Aerial / OS MasterMap):** download the tiles as GeoTIFF or JPG with a world file.
  They are already in EPSG:27700.
* **QGIS:** load the imagery (Project CRS = EPSG:27700). Then *right-click the layer → Export →
  Save As… → GeoTIFF*, with *Extent: map canvas* or a drawn box. Alternatively use
  *Project → Import/Export → Export Map to Image* and tick *Append georeference information*.
* 5–25 cm per pixel is ideal. Crop to the site; there's no need for the whole tile.

Good calibration features: kerb corners, road-marking ends, drain covers, bollard bases, paving
joints, lamp-post bases. The point is where the object **meets the ground**, never the top of a
post. Spread them over the **whole** area you care about, near and far, left and right. Don't put
them all in a line.

### 3.2 Set up the site

Your site configs live in `sites/` (kept out of git). `examples/` has commented ones for a
plaza, a motorway, and a per-clip file that extends a shared site config.

```powershell
groundtrack init plaza --template plaza          # writes sites/plaza.yaml
groundtrack init motorway --template motorway    # writes sites/motorway.yaml
```

Copy the video to `footage/plaza.mp4` and the GeoTIFF to `maps/plaza.tif`, or edit the paths
in the YAML. Paths are relative to the YAML file.

### 3.3 Pick the points

```powershell
groundtrack calibrate --config sites/plaza.yaml            # first frame
groundtrack calibrate --config sites/plaza.yaml --time 12  # or a frame with nobody in the way
```

A window opens with the video frame on the left and the map on the right.

1. **Click a point in the video**, then **the same point on the map**. Repeat for 6–8 pairs.
2. **Scroll** to zoom around the cursor; the toolbar's zoom and pan also work. **Right-click**
   or **u** undoes. **Enter** saves, **Esc** cancels.
3. From the 5th pair on, the title shows the live **RMSE** and the worst point. Yellow lines on
   the map go from where you clicked to where the fit puts that point.

You get a report like this:

```
  #        u        v            E            N   err (m)   LOO (m)  inlier
  1     38.5    313.4    529986.50    180013.60     0.076     0.141     yes
  ...
RMSE (inliers): 0.178 m   max: 0.345 m   leave-one-out RMSE: 0.371 m
Camera estimate (sanity check): 9.9 m above the ground at E 529999.4, N 179980.3; focal 1090 px
```

* **err** is the residual at each point; **RMSE** summarises them. The tool **warns if RMSE > 0.5 m**.
* **LOO** (leave-one-out) refits without that point and measures the error there. It's the more
  honest number with only 6–8 points. A point with a much bigger LOO than the others is either a
  bad click or the only point covering its area.
* **inlier = NO**: RANSAC rejected the point (more than 1 m off). Re-check that pair.
* **Camera estimate**: the camera height and position recovered from the homography. If you
  filmed from a 3rd-floor window (~9 m) and it says 25 m, the calibration is wrong even if the
  RMSE looks small.
* Far points are naturally less precise: at a low angle, one pixel can be 0.3 m on the ground.

Outputs go to `calibration/`:
* `plaza_homography.json`: the matrix, errors and camera estimate. Other tools can use `H_world`.
* `plaza_homography_points.csv`: your clicks. Re-running `calibrate` reloads them, so you can fix
  one point instead of starting over.
* `plaza_homography_check.png`: **open this.** The video frame is warped onto the map. Kerbs,
  markings and paving joints should line up. Anything above the ground (people, cars, walls)
  smears, and that's expected.

Non-interactive alternative: `groundtrack calibrate --config ... --points-csv my_points.csv --no-gui`
with columns `u,v,E,N` (for example ground control points you already have).

### 3.4 Optional: lens undistortion

Only worth doing if straight kerbs look visibly bent near the frame edges. Print a checkerboard,
and film or photograph it with the **same phone, lens and resolution** from 10–20 angles:

```powershell
groundtrack lens-calibrate --source checker_video.mp4 --pattern 9x6 --square-mm 25 --out calibration/phone_1x_lens.json
```

Set `lens: ../calibration/phone_1x_lens.json` in the site YAML, then **re-run `calibrate`**: the
picker then shows the undistorted frame. If the lens and homography don't match, the tool
refuses to run.

### 3.5 Optional: region of interest

```powershell
groundtrack roi --config sites/motorway.yaml
```

Click a polygon around the useful ground and press Enter. Leave out the far, blurry end of the
road, the sky, reflections, and parked cars you don't want. Detections whose **foot point** falls
outside are ignored before tracking. The polygon is saved as `sites/<site>_roi.json` and
picked up automatically. It is drawn on the calibration frame, so calibrate first.

### 3.6 If the camera moved: `stabilize`

A homography is only valid for the frame you clicked on. If the phone moved during the clip
(handheld, a railing that flexes, a slowly creeping tripod), set

```yaml
detection:
  stabilize: true
```

Every frame is then aligned to the calibration frame using features on the static
background (ORB + RANSAC; moving people and cars are ignored). Foot points are mapped back into
the calibration frame before projection. The calibration remembers which frame you clicked on
(`--frame` / `--time`), and the run log reports how far the camera moved and whether any frames
failed to align. Both templates have it on.

On the synthetic demo with a ~2° handheld-style sway, the camera drifts up to 68 px. Position
error is **0.14 m with `stabilize`** and **2.1 m without**, and speeds are off by 15 % without
it. Rotation (the usual handheld wobble) is compensated exactly. Walking around with the phone
is not: if the camera *position* moves by more than a few centimetres, film again from a fixed
spot.

Calibrate first, then track: tracking with `stabilize` aligns to the calibration frame. If you
re-calibrate on a different frame, re-run `track`.

---

## 4. Running both sites

```powershell
groundtrack run --config sites/plaza.yaml
groundtrack run --config sites/motorway.yaml
```

**Several clips from different camera positions?** Each position needs its own calibration
(and ROI). Keep the shared settings in one site file and add one small file per clip:

```yaml
# sites/motorway2.yaml
extends: motorway.yaml            # everything else comes from the shared site config
site: motorway2                   # -> runs/motorway2/...
video: "C:/.../Motorway2.mov"
homography: ../calibration/motorway2_homography.json
```

Then `groundtrack calibrate -c sites/motorway2.yaml`, `groundtrack roi -c sites/motorway2.yaml`
and `groundtrack run -c sites/motorway2.yaml`. Clips filmed from the *same* position can share a
homography file.

Each run writes a new folder, `runs/<site>/<YYYYmmdd-HHMMSS>/`. Detection is the slow part.
After changing cleaning, grid, stats or visual settings, re-run only the fast stages:

```powershell
groundtrack process --config sites/plaza.yaml --run latest      # or --run 20261007-153000
groundtrack track   --config sites/plaza.yaml --max-frames 300  # quick look at the first 10 s
```

Defaults that matter, with suggested values per site:

| setting | plaza | motorway | why |
|---|---|---|---|
| `detection.model` | `yolo26m.pt` | `yolo26m.pt` | `l`/`x` find more small, far objects but run slower |
| `detection.imgsz` | 1280 (1600–1920 for 4K) | 1280 | larger = smaller objects found |
| `detection.tracker` | `botsort` | `bytetrack` | both work; BoT-SORT is a little steadier in crowds |
| `detection.track_buffer_s` | 1.5 | 1.0 | how long a lost object is predicted before it's dropped |
| `groups.*.speed_range` | people 0–2.5 m/s | vehicles 0–35 m/s | colour ramp limits |
| `groups.*.smooth_window_s` | 1.0 | 0.6 | longer = smoother, but sharp turns get rounded |
| `cleaning.min_track_s` | 2.0 | 1.0 | drop flickers |
| `grid.cell_size_m` | 1.0 | 2.0 | field_grid resolution |
| `stats.count_lines` | entrances | one per carriageway | directional counts and flow/min |

Classes come from `groups`: each group lists its COCO classes (`person`, `bicycle`, `car`,
`motorcycle`, `bus`, `truck`) and carries its own physics (max plausible speed, smoothing,
ground offset, colour range). Remove a group to stop tracking it. Every option, with comments,
is in `groundtrack/config.py` (`DEFAULTS`).

### Vehicle ground point (read this for the motorway)

The bottom-centre of a car's box is where the car's footprint is **closest to the camera**. That
is not the car's centre. The tool offers two corrections:

* `offset_mode: travel` (as specified): shift the point **back along the direction of travel**
  by `ground_offset_m` (2.2 m by default; buses and trucks are set larger in the template). This
  is right for traffic **coming towards** the camera. For traffic **driving away**, the box
  bottom is the *rear* bumper, so this shifts the point the wrong way: about 4.4 m off in total.
* `offset_mode: view` (**recommended when you can see both carriageways**): shift the point
  **away from the camera** by the footprint's half-extent in that direction: half the length
  when seen end-on, half the width side-on, blended in between. It uses the camera position
  recovered from the homography, so it is correct for both directions and for curving roads.

The motorway template uses `travel`, as the brief asked. Switch it to `view` if you film both
directions.

### Machine-made gaps

Every row in `points.csv` has a `source`, and anything other than `detected` has `predicted=True`:

* `predicted`: the tracker's Kalman prediction while the object was hidden. Kept only if the
  object is found again; predictions trailing off after something leaves are thrown away.
* `stitched`: the tracker lost the object and later gave it a new ID. groundtrack re-linked the
  two fragments **in ground coordinates**, because the new one started where the old one was
  heading, within `stitch_radius_m`. Small, distant people behind columns often need this.
* `interpolated`: a single bad detection (a jump) that was removed and filled in.

In `topdown.png` these segments are white and dotted. `predicted_gaps.geojson` holds them as
separate lines for QGIS. `field_grid.csv` and `density.png` leave them out by default
(`grid.include_predicted`).

---

## 5. Outputs (one folder per run)

| file | contents |
|---|---|
| `points.csv` | `track_id, class, frame, time_s, x, y, vx, vy, speed, heading, predicted, source, group`. x/y in EPSG:27700 metres, v in m/s, heading in degrees clockwise from grid north |
| `track_summary.csv` | per track: duration, path length, straight-line distance, straightness, mean/median/max speed, start/end, numbers of predicted and stitched samples, original tracker IDs |
| `tracks.geojson` | one LineString per track (EPSG:27700) with the summary as attributes |
| `predicted_gaps.geojson` | only the machine-made stretches |
| `field_grid.csv` | world-aligned grid: `cell_x, cell_y, mean_vx, mean_vy, mean_speed, flow_speed, heading, coherence, count, n_tracks`. `count` = number of samples (your confidence); `coherence` is 1 when everyone moves the same way and 0 when flows cancel. `field_grid_<group>.csv` is written when there are several groups |
| `stats.json`, `stats.csv` | per class and group: tracks, flow per minute (as a time series and a mean), mean/median/85th-percentile speed, speed histogram, mean straightness, plus count-line crossings |
| `houdini_import.py` | Python SOP script (§6) |
| `topdown.png` | tracks over the dimmed GeoTIFF, coloured by speed blue → red, with a separate scale per group, scale bar and north arrow, 300 dpi |
| `topdown.mp4` | the same, animated: trails build up over the map with a dot at each current position, legend, scale bar and clock (`visuals.topdown_video`, `topdown_video_speedup`) |
| `density.png` | occupancy heat map: object-seconds per m² |
| `speed_histogram.png` | mean speed per track, same colours |
| `debug.mp4` | overlay on the original video, only with `--debug-video` (or `groundtrack debug-video`); people blurred unless `--no-blur`. Default **clean** style: only the tracks that survive cleaning, as smoothed trails coloured by speed with a dot at each current position, over a slightly dimmed picture that is aligned to the calibration frame (no camera shake). `--boxes` shows the full debug view (boxes, IDs, rejected tracks in grey), `--trail-s 5` keeps only the last 5 s. Colour = real m/s once calibrated, approximate m/s from body height before that |
| `raw_tracks.csv` | the tracker output in pixels (`bbox`, `confidence`, `predicted`) |
| `config_used.yaml`, `homography_used.json`, `run_log.txt` | provenance |

**QGIS:** drag `tracks.geojson` in; it carries its CRS. For `points.csv` or `field_grid.csv`, use
*Layer → Add Delimited Text Layer*, with X = `x` / `cell_x`, Y = `y` / `cell_y`, CRS EPSG:27700.
To draw arrows for the field grid, use the *Geometry Generator* or the *Vector Field Marker*
symbology with `mean_vx` / `mean_vy`.

---

## 6. Houdini import

1. Create a *Geometry* node. Inside it, add a **Python** SOP (*Python Script*).
2. Paste the contents of `runs/<site>/<run>/houdini_import.py` into its code box. Or paste this
   one-liner, which re-reads the file each cook:
   ```python
   exec(open(r"C:/path/to/runs/plaza/20261007-153000/houdini_import.py").read())
   ```
3. You get one point per sample with `track_id, class, group, frame, time_s, v` (velocity, m/s),
   `speed, heading, predicted` and `Cd`. There's one open polyline per track (with prim attribs
   `track_id`, `class`), and detail attribs `origin_E`, `origin_N`.

* **Axes:** Houdini is Y-up, so Easting → **+X** and Northing → **−Z**. North is up in the *Top*
  viewport. Coordinates are relative to the site origin (float32 can't hold BNG coordinates to
  the centimetre); add `origin_E/N` back for real-world positions.
* **Colour:** `Cd` uses the same blue → red ramp and per-group speed ranges as the PNGs (in linear
  colour). Predicted samples are darkened (`PREDICTED_DIM`).
* **Particles:** set `MODE = "current"` at the top of the script. You then get one point per track
  alive at the current time (`time_s = $T × TIME_SCALE + TIME_OFFSET`), with `v` set. Use it as a
  POP Source (emit from *all points*, *Use Inherited Velocity*) or in a POP Wrangle to steer agents.
  For trails, keep `MODE = "tracks"` and blast with `@time_s > $T`.

### Vector field for particle sims (`houdini_field.py`)

Every run also writes a **smoothed top-down vector field**:

* `vector_field.csv`: one row per grid cell (`field.cell_size_m`, aligned to British National
  Grid) with `vx, vy, speed, heading, density, weight, confidence`. It's a kernel-weighted
  average of the measured velocities (Gaussian, `field.smooth_m`). Speeds stay real, and the
  field fades out where there's no data instead of inventing motion there. With several
  groups you also get `vector_field_<group>.csv`, so cyclists don't speed up the pedestrian
  field.
* `vector_field_slices.csv`: the same field per `field.time_window_s` window (motorway default
  10 s), to animate traffic pulses.
* `flow_field.png`: streamlines of the field over the map; colour = speed, line width = how
  much data supports it.

In Houdini, make one Python SOP per role and set `MODE` at the top of `houdini_field.py`:

| MODE | gives you | use it for |
|---|---|---|
| `"volume"` | volumes `vel.x`, `vel.y`, `vel.z` (m/s), `density`, `confidence` | velocity field for POP Advect by Volumes |
| `"points"` | a point per cell with `v`, `speed`, `density`, `confidence`, `Cd`, `pscale` | arrows (Visualize `v`, or Copy to Points a line), guides, scattering |
| `"sources"` | points where tracks **start** (group `sources`) and **end** (group `sinks`) with `v`, `time_s`, `class` | emitters that match where people or vehicles really enter |

`GROUP = "people"` picks a group's field. `USE_TIME_SLICES = True` animates the field with `$T`.
`FADE_BY_CONFIDENCE` (on) scales velocity by confidence, so particles slow down gently at the
edge of the observed area instead of hitting a wall.

**A minimal particle sim**

1. `field`: Python SOP, `MODE = "volume"`.
2. `emit`: Python SOP, `MODE = "sources"`, then a Blast that keeps group `sources`. Or scatter
   points onto the `density` volume to emit where people actually are.
3. DOP Network → **POP Object** + **POP Source** (geometry from `emit`, *Emission Type: All
   Points*, a constant birth rate, *Initial Velocity: Use Inherited Velocity*) → **POP Advect by
   Volumes** (*Velocity Source: SOP*, path to `field`, *Velocity Volume*: `vel`, *Advection
   Type: Update Velocity* to follow the measured flow exactly, or *Update Force* for a looser,
   more organic follow) → optionally **POP Drag**.
4. Keep particles on the ground: POP Wrangle `@P.y = 0; @v.y = 0;`. Kill them where
   `confidence` is near 0 with a Volume Sample in a POP Wrangle.

If a node wants a VDB vector field: Convert VDB on the three `vel.*` volumes, then VDB Vector
Merge → `vel`. The axes and origin are the same as `houdini_import.py`, so the field, the
measured tracks and the particles all line up.

---

## 7. Ground-truth check

Measure the real accuracy at each site. The procedure takes five minutes.

1. Pick two ground points **A** and **B** 10–20 m apart, in the area you care about, with nothing
   in between. Read their coordinates in QGIS on the GeoTIFF (or tape-measure the length).
2. Film from the **exact same camera position**: stand still on A for 3 s, walk at a steady pace
   in a straight line to B, then stand still on B for 3 s.
3. Run:
   ```powershell
   groundtrack groundtruth --config sites/plaza.yaml --video footage/walk_test.mp4 `
       --start 529995.5 180005.0 --end 530004.5 180005.0
   # or without coordinates: --length 12.0   (length and speed only)
   # optional: --stopwatch 8.4   (walking time you measured), --track-id 3
   ```
4. Read the verdict, `groundtruth_report.json` and `groundtruth.png`:
   * **length error %**: the scale of the calibration (target within 3 %);
   * **speed error %**: the tool's speed against a reference that doesn't depend on the
     calibration: 80 % of the known length ÷ the video time between passing the 10 % and 90 %
     marks of the walk (or your `--stopwatch` time). Target within 5 %;
   * **start/end error**: absolute position while you stand on A and B (target < 0.5 m);
   * **lateral RMS**: wobble of the projected path around the straight line (target < 0.3 m).

On the synthetic demo (`groundtrack demo`, then `groundtrack groundtruth` on its video with A/B)
the tool scores 4–5 cm end-point error, -0.2 % length error, 0.1 % speed error and 5 cm lateral
RMS. Real footage will be worse; that's what this check is for.

---

## 8. Accuracy notes and limitations

* **Flat ground.** A homography maps one plane. Steps, ramps or a cambered road put objects off
  that plane, and they get projected as if they were on it. For a stepped plaza, calibrate on the
  main level and only use points on that level (or set the ROI to it).
* **Foot point.** A person's box bottom is their feet, which is good. A partly hidden person (a
  bollard covering their feet) gets a box that ends higher up, and is projected slightly *further
  away*. You can see this as a small kink just before an occlusion.
* **Partly hidden people.** When a railing, a sign or another person hides someone's legs, the
  box gets shorter from the bottom and the foot point jumps away from the camera. Each track
  keeps a robust normal box height; when a box is clearly shorter and its *bottom* edge is the
  one that moved, the feet are rebuilt as top + normal height (`recover_occluded_feet`). If the
  *top* moved instead (a gantry, an umbrella), the real bottom is kept. On a crowded plaza clip
  this repairs ~7 % of all boxes.
* **Boxes cut off by the bottom of the frame** are dropped, because their "feet" are really
  their waist (`edge_margin_px`).
* **Parked vehicles** are dropped: vehicles that never move more than `min_displacement_m`
  (5 m). Standing people are kept on purpose, because dwelling is data.
* **People on bicycles** are dropped as `person` when they sit on a detected bicycle
  (`suppress_riders`), so they are counted once, as a bicycle.
* **Speeds** come from the derivative of the smoothed path, so they are as good as the
  calibration scale. Check with §7. Headings are relative to **grid north**, which differs from true north
  by up to a few degrees depending on how far east or west the site is (zero at 2° W).
* **Field grid and density are time-weighted.** A slow walker leaves more samples per cell than a
  runner, which is what you want for occupancy, not for counts. For counts, use `count_lines`.

---

## 9. Troubleshooting

| symptom | fix |
|---|---|
| `groundtrack device` says cpu on the NVIDIA laptop | you installed the CPU wheel: `pip uninstall torch torchvision` and reinstall from the `cu128` index (§1); update the NVIDIA driver |
| calibration RMSE > 0.5 m | spread the points out, use ground-contact features, zoom in before clicking, check the map really is EPSG:27700, film with **stabilisation off** and the **1× lens** |
| check image lines up at the centre but not at the edges | lens distortion: use the 1× lens, or run `lens-calibrate` (§3.4) |
| speeds drift over a long recording | the camera moved (tripod knocked, window flexing); re-calibrate on a frame from that part |
| ghost tracks over windows / glass | **avoid window reflections**: lens against the glass, dark cloth, room lights off; mask with an ROI |
| many short broken tracks | increase `imgsz` or use a bigger model; raise `track_buffer_s`; check `stitch_radius_m`; set an ROI to cut the blurry far field |
| car/truck/bus labels flicker | handled: each track gets a confidence-weighted majority class |
| `Video is 3840x2160 but the homography was calibrated on 1920x1080` | calibrate on a frame of the same footage settings |
| speeds about 2× or 0.5× too high or low | the phone recorded at variable fps or the container lies: set `fps:` in the config, and turn off Auto FPS |
| slow on a laptop | `vid_stride: 2` (halves the work; speeds are still right), `yolo26s.pt`, ROI |
| picker window doesn't open | needs a desktop session (Tk); over SSH, use `--points-csv ... --no-gui` |

---

## 10. Privacy

* Only bounding-box coordinates are stored. **No face crops, no images of people, no
  appearance embeddings, no identities.** ReID is off, and track IDs are arbitrary numbers that
  are only meaningful within one video.
* `debug.mp4` is **off by default**. When you do render it, people are **blurred** unless you pass
  `--no-blur`. Don't share an unblurred debug video.
* **Delete the raw footage once you've processed it and checked the run.** `points.csv`,
  `tracks.geojson` and the rest contain everything the research needs. In PowerShell:
  `Remove-Item footage\plaza.mp4`. On macOS: `rm footage/plaza.mp4`, then empty the Bin. Also
  delete the copies on your phone and in its cloud backup (iCloud / Google Photos "Recently
  deleted"), and any `debug.mp4`. Keep the calibration files: they hold no personal data.
* If the footage might show identifiable people, follow your institution's ethics and data
  protection guidance (UK GDPR). Signage at the site and short retention periods are standard
  practice.

---

## 11. Tests

```powershell
pytest                 # everything, including the end-to-end run (~40 s on the GPU)
pytest -m "not e2e"    # unit tests only (~2 s)
```

* `test_homography.py`: exact recovery on a synthetic camera, click noise, a rejected bad click,
  the RMSE warning, the horizon, the save/load round trip.
* `test_trajectories.py`: smoothing and velocity on a straight line and a circle, heading hold,
  spike removal, splitting at an ID switch, vehicle offsets (both modes), camera recovery from
  the homography, end-to-end `process_tracks` on synthetic boxes, the class majority vote.
* `test_exports.py`: grid aggregation and alignment, GeoJSON validity (including reading it with
  GDAL), the predicted-gap GeoJSON, count-line directions, the stats table. The Houdini script is
  run against a stub `hou` module.
* `test_e2e.py`: generates the demo video, runs calibration, YOLO, tracking and all exports, then
  checks positions (within 25 cm), speeds (within 6 %), the occlusion bridging, and the
  `groundtruth` command.

---

## License

AGPL-3.0, matching [Ultralytics](https://github.com/ultralytics/ultralytics), whose YOLO models
and trackers this tool builds on. See `LICENSE`.
