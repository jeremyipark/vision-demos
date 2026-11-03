# How it works

Every threshold is in [`config.py`](config.py).

## Shots

ViTPose poses the shooter on every frame. A shot is both wrists over the head,
after both were below the shoulders.

## Hoop

The phone stays still, so SAM 3.1 finds the hoop once, on one frame. The
prompt "basketball hoop" returns the rim and net. The rim is the widest part of
that mask, and its bottom is the bottom of the net.

## Ball

SAM tracks the ball only around each shot, from 0.6 s before to 2.4 s after,
every frame. If the ball is still at the rim when a window ends, tracking runs
1.5 s longer. Dribbling between shots is never tracked.

In the hands and the net SAM's box grows past the ball, so boxes that aren't
ball-sized and round are dropped.

## Make or miss

From one camera, a ball falling just in front of or behind the rim can look like
it went in. So a make has to do five things, in order:

1. **Come down from above the rim.** Rules out an airball into the net.
2. **Cross the rim line inside the rim** (its middle 80%).
3. **Be at the rim, not in front of it.** At the rim's distance the ball is about
   half the rim's width (0.24 m ÷ 0.49 m). A ball more than 1.2× that size is
   closer to the camera, falling short in front of the rim.
4. **Leave the net slowed by it.** Rim to net bottom takes 0.2 s or more through
   the net, about 0.1 s falling past it. The cutoff is 0.15 s.
5. **Never rise back over the rim.** Rules out an in-and-out.

The rim and net often hide the ball as it goes through, so a crossing can be
judged across a gap of up to 0.5 s.

Every downward crossing is checked, so a ball that bounces on the rim and drops
in still counts. A miss that bounced at the rim is labelled "off the rim".

`plots.png` draws every ball path against the rim, so each call can be checked.

## Hot streak

After 3 makes in a row (`HOT_STREAK`), an animated "HOT STREAK!" shows in the
video's top-right corner, fading while the ball passes behind it. The next shots
leave a short flame behind the ball, and the rim flares on each make, then cools
to a warm glow. Each further make runs hotter. A miss resets it.

## Court

No court lines are in shot, so the spots are estimated. The camera is treated
as a pinhole with a 70° field of view. The rim (0.49 m across its outer edge,
which is what the mask spans) gives the hoop's distance, and the shooter's size
gives theirs (shoulder to ankle ≈ 0.78 × height). Each one's position in the
frame gives the sideways offset.

The shooter's size is taken standing tall, over the 2 s around the shot. Bent
knees in the gather look shorter, which would read as farther away.

Left and right are fairly reliable. Distance from the hoop scales with `--fov`
and `--player-height`.

The court is drawn with only the lines that matter here: the sidelines, the
three-point line, and the paint (the middle third of the three's width) up to
the free-throw line.

## Elbow

The elbow angle is the angle between the upper arm and forearm of the shooting
arm. It is measured at the **set point** (the deepest bend) and at the
**release** (the fullest extension just after the ball leaves).

It is measured in 2D, so only side-on shots count: shoulder width under 0.55 of
the torso length. Facing the camera or facing away, the arm points at the lens
and the angle is meaningless.

`elbow.png` lines each side-on shot up on its release. Beside it, a strip of
frames per shot shows the arm and its angle (`ELBOW_STRIP` in `config.py`).

## On the video

The top half is the clip with the pose, the ball and its path, and the rim,
which lights green or red as each shot is decided. The bottom half, the same
height, is the court with a dot per shot and field goals and FG% in its corner.

The ball is drawn for the whole shooting motion, from its lowest point in the
hands: the hip for a shot from the pocket, the shoulder for a catch-and-shoot.
If the body hides it there, it shows once it comes into view. Drawing uses a
looser pick of the ball than judging, and fills gaps of up to a third of a
second, so the ring glides rather than blinks.

## Limits

- One shooter, one ball, a fixed camera.
- Shots are found from the shooter's pose, so a shot taken out of frame isn't counted.
- The elbow needs a side-on view; makes and misses don't.
- Court spots are an estimate, not a measurement.
