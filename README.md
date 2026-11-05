# vision-demos

Real-world computer vision demos.

| Project | What it does | Key model |
|---|---|---|
| **[dance_sync](dance_sync/)** | Compares dancers' sync performing the same choreography and computes a similarity metric. | [`vitpose-plus-large`](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large) |
| **[chin_ups](chin_ups/)** | Counts chin-up reps from a clip and times the ascent and descent of each one. | [`vitpose-plus-large`](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large) |
| **[pull_ups](pull_ups/)** | Counts pull-up reps, times each pull, and grades each rep's range of motion as full or partial against the athlete's own calibrated dead hang. | [`vitpose-plus-large`](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large) |
| **[push_ups](push_ups/)** | Counts push-up reps from a side-on clip, times each push, and grades each rep for full range of motion and a straight back. | [`vitpose-plus-large`](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large) |
| **[rock_climbing](rock_climbing/)** | Segments bouldering holds, returns which holds the climber used and in what order, and compares attempts at the same route. | [`sam3.1`](https://vlm.run/gateway/models/facebook-sam3.1) + [`vitpose-plus-large`](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large) |
| **[rock_climbing_3d](rock_climbing_3d/)** | Reads the same route from a video recorded with depth (iPhone LiDAR), placing the wall, holds and climber in 3D and measuring the route in meters. | [`sam3.1`](https://vlm.run/gateway/models/facebook-sam3.1) + [`vitpose-plus-large`](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large) |
| **[running](running/)** | Measures a runner's cadence, times every foot strike, and averages the knee shape at contact. | [`vitpose-plus-large`](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large) |
| **[deadlift](deadlift/)** | Counts deadlift reps and classifies each rep's back as straight or rounded, with the probability. | [`sam3.1`](https://vlm.run/gateway/models/facebook-sam3.1) + [`vitpose-plus-large`](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large) + [`gemma-4-26b-a4b-it`](https://vlm.run/gateway/models/google-gemma-4-26b-a4b-it) |
| **[basketball](basketball/)** | Calls every shot a make or a miss by following the ball through the rim, and plots each shot on a court from above with the FG%. | [`sam3.1`](https://vlm.run/gateway/models/facebook-sam3.1) + [`vitpose-plus-large`](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large) |


### Rock Climbing 3D

<p align="center">
  <a href="https://www.youtube.com/watch?v=vjItC11jF0o"><img src="https://img.youtube.com/vi/vjItC11jF0o/maxresdefault.jpg" width="600" alt="A climbing route reconstructed in 3D from iPhone LiDAR, with the holds used numbered in order and the route measured in meters. Click to watch on YouTube."></a>
  <br>
  <a href="https://www.youtube.com/watch?v=vjItC11jF0o">▶ Watch on YouTube</a>
</p>

### Dance Sync

<p align="center">
  <a href="https://www.youtube.com/watch?v=Hmfwv0DwPeA"><img src="https://img.youtube.com/vi/Hmfwv0DwPeA/maxresdefault.jpg" width="600" alt="Dancers with pose overlays on the left and a sync score panel on the right. Click to watch on YouTube."></a>
  <br>
  <a href="https://www.youtube.com/watch?v=Hmfwv0DwPeA">▶ Watch on YouTube</a>
</p>

### Chin-Ups

<p align="center">
  <a href="chin_ups/"><img src="chin_ups/readme_images/chin_ups_demo_thumbnail.jpg" width="600" alt="A chin-up at the top of the rep with a pose overlay on the left and a rep-timing panel on the right"></a>
</p>

### Pull-Ups

<p align="center">
  <a href="https://www.youtube.com/watch?v=yAoVkH60Exw"><img src="https://img.youtube.com/vi/yAoVkH60Exw/maxresdefault.jpg" width="600" alt="A six-rep pull-up set with a pose overlay on the left and, on the right, vertical displacement over time against the dead hang and the bar, and each rep graded full or partial. Click to watch on YouTube."></a>
  <br>
  <a href="https://www.youtube.com/watch?v=yAoVkH60Exw">▶ Watch on YouTube</a>
</p>

### Push-Ups

<p align="center">
  <a href="push_ups/"><img src="push_ups/readme_images/push_ups_demo_thumbnail.jpg" width="600" alt="A push-up set with a pose overlay on top and a live panel below: the rep count, 23 of 23 full range, 22 of 23 straight back, vertical displacement, the body line, and the push-up time per rep"></a>
</p>

### Rock Climbing

<p align="center">
  <a href="rock_climbing/"><img src="rock_climbing/readme_images/rock_climbing_demo_thumbnail.jpg" width="600" alt="A completed boulder problem: the left panel holds a card comparing this attempt's sequence of holds against three others, the right panel the segmented route with the holds used lit in order"></a>
</p>

### Basketball

<p align="center">
  <a href="https://www.youtube.com/shorts/TgUP5C8-8NE"><img src="https://img.youtube.com/vi/TgUP5C8-8NE/maxresdefault.jpg" width="600" alt="The 10th shot just after release, with pose, ball and rim overlay; below, the shot chart with 7 of 9 field goals and 78% in its corner. Click to watch on YouTube."></a>
  <br>
  <a href="https://www.youtube.com/shorts/TgUP5C8-8NE">▶ Watch on YouTube</a>
</p>

### Running

<p align="center">
  <a href="running/"><img src="running/readme_images/running_demo_thumbnail.jpg" width="600" alt="A runner on a treadmill with a pose overlay on the left and a live cadence panel on the right"></a>
</p>

### Deadlift

<p align="center">
  <a href="https://www.youtube.com/watch?v=917J5oOvcBM"><img src="https://img.youtube.com/vi/917J5oOvcBM/maxresdefault.jpg" width="600" alt="A four-rep deadlift set with pose and plate overlay on the left and the live back verdict, bar height and per-rep results on the right. Click to watch on YouTube."></a>
  <br>
  <a href="https://www.youtube.com/watch?v=917J5oOvcBM">▶ Watch on YouTube</a>
</p>

## Run

Each project has its own README with setup and instructions. To try them in a browser instead:

```bash
conda env create -f environment.yml && conda activate vision-demos   # or: pip install -r requirements.txt
python app.py                                                        # -> http://127.0.0.1:7860
```

Upload a clip, trim it, analyze. No API key needed (anonymous, rate-limited tier); paste a [VLM Run](https://app.vlm.run) key or set `VLMRUN_API_KEY` for higher limits. `rock_climbing_3d` (it needs a LiDAR capture) and `push_ups` aren't in the app. ffmpeg is required.

## License

[Apache-2.0](LICENSE).
