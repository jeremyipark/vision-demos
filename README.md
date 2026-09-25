# vision-demos

Real-world computer vision demos.

| Project | What it does | Key model |
|---|---|---|
| **[dance_sync](dance_sync/)** | Compares dancers' sync performing the same choreography and computes a similarity metric. | [`vitpose-plus-large`](https://docs.vlm.run/gateway/models/usyd-community-vitpose-plus-large) |
| **[chin_ups](chin_ups/)** | Counts chin-up reps from a clip and times the ascent and descent of each one. | [`vitpose-plus-large`](https://docs.vlm.run/gateway/models/usyd-community-vitpose-plus-large) |
| **[rock_climbing](rock_climbing/)** | Segments bouldering holds, returns which holds the climber used and in what order, and compares attempts at the same route. | [`sam3.1`](https://docs.vlm.run/gateway/models/facebook-sam3.1) + [`vitpose-plus-large`](https://docs.vlm.run/gateway/models/usyd-community-vitpose-plus-large) |
| **[rock_climbing_3d](rock_climbing_3d/)** | Reads the same route from a video recorded with depth (iPhone LiDAR), placing the wall, holds and climber in 3D and measuring the route in meters. | [`sam3.1`](https://docs.vlm.run/gateway/models/facebook-sam3.1) + [`vitpose-plus-large`](https://docs.vlm.run/gateway/models/usyd-community-vitpose-plus-large) |
| **[running](running/)** | Measures a runner's cadence, times every foot strike, and averages the knee shape at contact. | [`vitpose-plus-large`](https://docs.vlm.run/gateway/models/usyd-community-vitpose-plus-large) |


### Rock Climbing 3D

<p align="center">
  <a href="https://www.youtube.com/watch?v=vjItC11jF0o"><img src="https://img.youtube.com/vi/vjItC11jF0o/maxresdefault.jpg" width="600" alt="A climbing route reconstructed in 3D from iPhone LiDAR, with the holds used numbered in order and the route measured in meters. Click to watch on YouTube."></a>
  <br>
  <a href="https://www.youtube.com/watch?v=vjItC11jF0o">▶ Watch on YouTube</a>
</p>

### Dance Sync

<p align="center">
  <a href="dance_sync/"><img src="dance_sync/readme_images/dance_demo_thumbnail.jpg" width="600" alt="Dancers with pose overlays on the left and a sync score panel on the right"></a>
</p>

### Chin-Ups

<p align="center">
  <a href="chin_ups/"><img src="chin_ups/readme_images/chin_ups_demo_thumbnail.jpg" width="600" alt="A chin-up at the top of the rep with a pose overlay on the left and a rep-timing panel on the right"></a>
</p>

### Rock Climbing

<p align="center">
  <a href="rock_climbing/"><img src="rock_climbing/readme_images/rock_climbing_demo_thumbnail.jpg" width="600" alt="A completed boulder problem: the left panel holds a card comparing this attempt's sequence of holds against three others, the right panel the segmented route with the holds used lit in order"></a>
</p>

### Running

<p align="center">
  <a href="running/"><img src="running/readme_images/running_demo_thumbnail.jpg" width="600" alt="A runner on a treadmill with a pose overlay on the left and a live cadence panel on the right"></a>
</p>

Each project has its own README with setup and instructions.

## License

[Apache-2.0](LICENSE).
