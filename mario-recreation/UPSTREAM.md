# Provenance

Base engine: [Gold872/Super-Mario-Bros](https://github.com/Gold872/Super-Mario-Bros), commit `609c5af217abf6eb55d3eb594df4db76cfb0c90c`, retrieved 2026-11-20. Its source and game resources are vendored under `vendor/`; the original MIT license is preserved in `vendor/LICENSE`. This project adapts that recreation rather than claiming it as newly authored code.

Local changes include:

- Portable CMake build and a new native executable entry point.
- 16-tile-wide NES viewport, pixel-aligned rendering, bitmap HUD, fixed-resolution frame capture.
- Main-thread level construction, a seeded MT19937 generator, process-isolated resets, synchronous action stepping, JSON state, and RGB recording.
- Linear walking/running acceleration and speed-dependent variable jumps informed by the original physics tables; this remains a floating-point collision engine.
- Coin/timer scoring, coin rollover, fireball count limit, and flag score.
- ECS iteration/component ownership and command-queue safety fixes needed for long deterministic rollouts.
- Corrected World 1-1's opening cloud position against the NES map.

Primary technical reference: [SMB disassembly, physics tables and routines](https://github.com/MrWint/smb-dis/blob/master/smbdis.asm). The local changes use the documented numeric movement parameters, not an emulated 6502 or a cycle-accurate port. The disassembly is not bundled.

Visual references:

- [HappyLee's Super Mario Bros. walkathon TAS recording](https://archive.org/details/supermariobros_walkathonv3_happylee). Downloaded `supermariobros-walkathonv3-happylee.avi` for extracting comparison frames. It is an actual NES game recording with a tool-assisted input sequence, and includes an encoder credit overlay near the beginning. The video is ignored by version control; selected PNG frames are retained for the local validation report.
- [NESMaps World 1-1 background map](https://www.nesmaps.com/maps/SuperMarioBrothers/SuperMarioBrosWorld1-1MapBG.html). The PNG is a reference for tile geometry and palette, not a gameplay video.
- [Nintendo's original instruction booklet](https://www.nintendo.co.jp/clv/manuals/en/pdf/CLV-P-NAAAE.pdf).

The graphics, audio, character designs, and original game are Nintendo material. Upstream's code license is not an asset license. The upstream font is retained with the original resources; gameplay HUD text is rendered by `native/BitmapFont.h`.
