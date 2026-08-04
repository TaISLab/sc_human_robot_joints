# Participant parameter files

This folder holds one YAML file per volunteer (`P01.yaml`, `P02.yaml`, ...),
following the schema in [`schema/participant_template.yaml`](schema/participant_template.yaml).

## Privacy

Participant files must contain **only** an anonymized ID and physical/
kinematic parameters -- never a name, contact detail, or any other directly
identifying field. The name-to-ID mapping is recorded separately, on paper,
outside this repository (and outside any digital system), under the
approved (and, per `paper_ral.tex` Sec. 5.3, soon-to-be-amended) ethics
protocol.

Because a correctly-filled file carries no identifying information by
construction, `participants/P*.yaml` files **are** synced to this
repository -- unlike raw video/image data, which is never committed here
(see below). Before adding or committing a new participant file, confirm:

1. The file contains no identifying information (re-read the checklist
   above) -- only the anonymized ID and physical/kinematic fields defined
   in the schema. Do not add free-text fields (e.g. `notes`) that could
   describe the participant in an identifying way.
2. The anonymized ID matches the one on the paper record, and that paper
   record -- not this repository -- is what anyone would need to re-link a
   file to a person.
3. If/when this repository is made public (currently planned for after
   acceptance, see root `README.md`), double-check that the informed
   consent actually covers releasing these de-identified parameters
   publicly, even though they carry no direct identifiers -- ask before
   flipping the repo's visibility, not after.
4. No raw video or image data is included anywhere in this repository --
   videos/images from the visuo-tactile workstation must stay in a
   separate, access-controlled location, consistent with the IEEE RAS
   double-anonymous guidelines' instruction to blur faces in any video/
   image material, and with standard human-subject data handling.

## Adding a new participant

```bash
cp participants/schema/participant_template.yaml participants/P01.yaml
# fill in participant_id, anthropometry, joint_limits_rad, session
```

## Relationship to the codebase

- `anthropometry.upper_arm_length` / `forearm_length` correspond directly to
  `config/shared_control.yaml`'s `upper_arm_length` / `forearm_length`
  (same units, same meaning) -- a participant file is meant to override
  those defaults for a specific session, not replace the config file.
- `joint_limits_rad` follows the same `[q_min, q_max]` per-joint convention
  and ordering as `performance.DEFAULT_JOINT_LIMITS` in
  `shared_control_pkg/performance.py`.
- `study_arm` is always `"right"` given the current single-arm tracking
  limitation (see `paper/paper_ral.tex`, Sec. 4.8, "Single-arm
  (right-arm-only) tracking").
