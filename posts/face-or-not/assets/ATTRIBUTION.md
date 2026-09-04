# Asset attribution

`dataset_samples.png` contains deterministic examples from the published dataset's training split. No validation or test image is used. Open Images lists each source image under [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/). The crop annotations are from Open Images and are licensed CC BY 4.0; see the dataset card for the source caveat and complete build method.

| # | label | title | author | source | license | Open Images ID |
|---:|---|---|---|---|---|---|
| 1 | no_face | YuuuuuuuuuuuuuuuuuuuuuM | [Mayesha k](https://www.flickr.com/people/synth/) | [original](https://www.flickr.com/photos/synth/91921496) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) | `775efcd4846a7744` |
| 2 | no_face | Warriors | [nuklr.dave](https://www.flickr.com/people/david_mackay/) | [original](https://www.flickr.com/photos/david_mackay/5801235882) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) | `322ea1e3774ecef6` |
| 3 | no_face | Leffe | [J B](https://www.flickr.com/people/lobsterstew/) | [original](https://www.flickr.com/photos/lobsterstew/3644390118) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) | `3ba5380490084697` |
| 4 | no_face | ムーミン谷のなかまッフル | [Kentaro Ohno](https://www.flickr.com/people/inucara/) | [original](https://www.flickr.com/photos/inucara/4414824150) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) | `578cb03cb6c23799` |
| 5 | face | Miss IQ&Beauty (3) | [Piotr Drabik](https://www.flickr.com/people/drabikpany/) | [original](https://www.flickr.com/photos/drabikpany/5471756852) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) | `72d394241bbaba93` |
| 6 | face | Boston Jazz Festival 2007 023 | [Bryan Maleszyk](https://www.flickr.com/people/maleszyk/) | [original](https://www.flickr.com/photos/maleszyk/1469850879) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) | `acd8371d50566145` |
| 7 | face | TFD (The First drop) Open Day | [Pete Bellis](https://www.flickr.com/people/video4net/) | [original](https://www.flickr.com/photos/video4net/4102045455) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) | `e07438cdd4d9165a` |
| 8 | face | brittney | [Argyleist](https://www.flickr.com/people/bnimble/) | [original](https://www.flickr.com/photos/bnimble/1189984696) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) | `16327bd5c199d0f7` |

The machine-readable ledger is [`attribution.jsonl`](./attribution.jsonl). Its `pixel_sha256` hashes the decoded 128x128 RGB crop bytes, and `asset_box_px` locates that crop inside the generated grid.
