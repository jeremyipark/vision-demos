# basketball

Tracks a shooting session from a phone that stays still. It calls every shot a **make
or a miss** by following the ball through the rim, and plots each shot on a
**court seen from above** with a running field-goal percentage. Three makes in a
row set the ball and rim on fire. It also graphs the **shooting elbow** for
side-on shots.

Pose runs on
[`vitpose-plus-large`](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large),
the hoop and ball on [`sam3.1`](https://vlm.run/gateway/models/facebook-sam3.1),
both through the [VLM Run Gateway](https://www.vlm.run/gateway). How it works:
[how-it-works.md](how-it-works.md).

<p align="center">
  <a href="https://www.youtube.com/shorts/TgUP5C8-8NE"><img src="https://img.youtube.com/vi/TgUP5C8-8NE/maxresdefault.jpg" width="600" alt="The 10th shot just after release, with pose, ball and rim overlay; below, the shot chart with 7 of 9 field goals and 78% in its corner. Click to watch on YouTube."></a>
  <br>
  <a href="https://www.youtube.com/shorts/TgUP5C8-8NE">▶ Watch on YouTube</a>
</p>

## Run it

1. Get an API key at [app.vlm.run](https://app.vlm.run/sign-in) and put it in a `.env` at the repo root:

   ```bash
   cp ../.env.example ../.env   # then paste your key after VLMRUN_API_KEY=
   ```

2. Create the environment:

   ```bash
   conda env create -f environment.yml
   conda activate basketball
   ```

3. Put a clip in `data/input/`: landscape, the phone set down and still, with the hoop and the shooter in shot.

4. Run it:

   ```bash
   python main.py                         # every clip in data/input/
   python main.py data/input/session.mov  # one clip
   ```

   | flag | sets |
   |---|---|
   | `--height 1080` | the output's total height; the clip and the court below each take half, up to the source's own resolution |
   | `--hand right` | the shooting hand (default: auto) |
   | `--player-height 1.75` | the shooter's height in meters; scales court distances |
   | `--fov 70` | the camera's horizontal field of view |
   | `--yaw 0` | turns the court if the camera isn't square to the backboard |
   | `--three 6.75` | three-point radius: 6.75 FIBA, 7.24 NBA, 6.02 high school |
   | `--trim 10` | only the first 10 s |
   | `--fresh` | ignore cached replies |
   | `--li` `--x` `--ig` / `--no-credit` | a credit in the corner |

   Everything else is in [`config.py`](config.py).

## Output

Each run writes to `data/output/<timestamp>-shot-chart/`:

| file | what |
|---|---|
| `<clip>_basketball.mp4` | the video on top, field goals and the shot chart below |
| `plots.png` | every ball path at the hoop, and the shot chart |
| `elbow.png` | the elbow through each side-on shot, with a strip of frames per shot |
| `<clip>_shots.csv` | one row per shot |
| `<clip>_frames.csv` | one row per frame |
| `shots.json`, `run.json` | per-shot results with ball paths; settings and cost |

Model replies are cached in `data/cache/`, so re-runs don't call the models again.

## Helper scripts

Optional extras, run from this directory on the clips in `data/input/` (they reuse
`main.py`'s cached replies; add `--cached-only` to skip the gateway).

- `python history.py`: charts and CSVs across all your sessions.
- `python highlights.py`: a highlight reel of every shot, misses first.
- `python social.py`: a shareable video of a session's shots.
- `python compilation.py`: a mixed reel of swishes, rim-then-ins and misses.
- `python rim.py`: swish, or where the ball hit the rim.
- `python trajectory.py`: ball arcs, makes vs misses.
- `python form.py`: elbow and knee form, makes vs misses.

`social.py` writes to `data/output/<timestamp>-swish-detection/`; the others to
`data/output/<script>/`. Each script's docstring lists its flags.

### Swish detection

`social.py` also makes a shareable video of a session's shots, with the clip on top
and **Shot pathway** and **Shot outcome** panels below. Each shot's ball path is drawn
at the hoop and marked where it met the rim; shots are tallied as swish, make with
rim, or miss.

<p align="center">
  <img src="readme_images/swish_detection_thumbnail.jpg" width="400" alt="The 7th shot just after release, with pose and ball overlay; below, ball paths at the hoop and bars for 2 swishes, 2 makes with rim and 2 misses">
</p>

## License

[Apache-2.0](../LICENSE).
