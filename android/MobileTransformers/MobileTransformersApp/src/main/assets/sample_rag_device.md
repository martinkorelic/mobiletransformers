# Heat, battery and when work actually runs

## Thermal throttling

A phone has no fan. Sustained compute raises the die temperature until the governor reduces clock
speeds, so a long training run gets slower the longer it lasts — the first few hundred steps are not
representative of the last few hundred. The SDK samples the thermal status and pauses when the device
reports a severe state, which is slower than pushing through but avoids the system killing the app
outright.

## Charging and battery

Scheduled training is constrained to run while charging by default. This is not only about battery
level: charging usually means the phone is stationary and unattended, which is exactly when a
multi-minute compute job is acceptable. A run started on battery competes with whatever the user is
actually doing.

## Doze, and why a start time is a floor

Android batches deferrable background work and can hold it during Doze, so a scheduled run promises a
*minimum* delay rather than an appointment. Asking for a start in fifteen minutes means "not before
fifteen minutes", and the actual start may be considerably later if the screen is off and the device
is idle. An exact wall-clock start would require the exact-alarm permission, which the Play Store
restricts to alarm clocks and calendar reminders — so the honest design is to state the limitation
rather than work around it.

## Memory

Model weights are memory-mapped rather than read into the heap, so the size of a package on disk is a
poor predictor of whether it will run. A three-and-a-half gigabyte inference directory works on a
device with two and a half gigabytes available, because the pages are backed by the file rather than
by the heap. Training is different: the optimizer state and the activations are genuinely allocated,
and that is where a run fails.

## Foreground work

Long-running training runs as a foreground service with a persistent notification showing the current
step and loss. This is mandatory rather than decorative — Android will not let a background process
hold the CPU for minutes at a time, and a visible notification is the price of being allowed to.
