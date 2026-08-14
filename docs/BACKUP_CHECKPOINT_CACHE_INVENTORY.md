# Backup Inventory — Checkpoint & Cache Metadata (extracted 13 Aug 2026)

> **Purpose**: permanent machine-readable record of every `.pth` checkpoint and `.npz` probability cache that was backed up on 12 Aug 2026, so the heavy backup zips (`06_models_pth_all.zip`, `07_caches_npz_all.zip`) can be removed from disk **without losing knowledge of what we had**. Each file's SHA-256 is retained so any regenerated artifact can be verified byte-for-byte.
> **Status**: 06 and 07 zips deleted after this record was written. All 9 required checkpoints remain in the working tree. All `.npz` caches are regenerable by their notebooks.

## Regeneration reference

- Full per-artifact regeneration recipes (which notebook/script recreates each checkpoint/cache): `docs/REPRODUCTION_AND_REGENERATION.md`.
- The 9 required input checkpoints are **still present** in the working tree (see §3 below) — deleting the backup zip removed only their redundant copies.

## 1. Checkpoints — `06_models_pth_all.zip` (was 5,767 MB / 80 files)

| File | Size | SHA-256 (verify regeneration) |
|---|---|---|

### Practice/multitask_liver_tumor_outputs

| `Practice/multitask_liver_tumor_outputs/multitask_best.pth` | 82.4 MB | `9c4160bbd68891f9dc4e5f04ceca4391f38c5869b3f81c72b95d4639e0572223` *(still in working tree — backup copy redundant)* |
| `Practice/multitask_liver_tumor_outputs/multitask_last.pth` | 82.4 MB | `f506e4ca0b561cc0f1050089c16da42c01b87af76fa55ea213a3b61123d0ca9f` |

### Practice/patient_aware_baseline_outputs

| `Practice/patient_aware_baseline_outputs/patient_aware_best.pth` | 82.4 MB | `10b7653f2085f50ad730bbeb1aa924b008357d146b2a839b63eccead3b647d11` |
| `Practice/patient_aware_baseline_outputs/patient_aware_last.pth` | 82.4 MB | `03e9c3d8d930688bfb53371a539f093684fdbeb8d2c9e493b58c9630ab9cad92` |

### Practice/recall_aware_loss_outputs

| `Practice/recall_aware_loss_outputs/patient_aware_best.pth` | 82.4 MB | `0ee8d739165726dabd5ac15aeaa8a0cdf61d06724749e0f6b9f61a92c1b337fc` |
| `Practice/recall_aware_loss_outputs/patient_aware_last.pth` | 82.4 MB | `b9196239d1c3c760122bbf92a500b78d2a81e2b2b7b22389218818d39588781f` |

### Practice/stabilized_composite_loss_outputs

| `Practice/stabilized_composite_loss_outputs/patient_aware_best.pth` | 82.4 MB | `4aacee3a88a070fbe54e590de857e123af1dc95ffdda2b174e31ab99cfc5266b` |
| `Practice/stabilized_composite_loss_outputs/patient_aware_last.pth` | 82.4 MB | `3270ebfe40dad6d2b9f10e410596c6d9429dbca9cc2012a08a40a9432b93f31a` |

### Practice/patient_lesion_balanced_outputs

| `Practice/patient_lesion_balanced_outputs/patient_aware_best.pth` | 82.4 MB | `1204a14abd4961cdb0c807659cc2310c72d7f499bc6969143db72c8cd52053e9` |
| `Practice/patient_lesion_balanced_outputs/patient_aware_last.pth` | 82.4 MB | `f9f4b584b155bfda4c8468aa0f65325ec61c705da20c733f14c1165339fac5d5` |

### Practice/context_2_5d_outputs

| `Practice/context_2_5d_outputs/patient_aware_best.pth` | 82.4 MB | `264436f825ef18979dc34195c0b5f5d855d9a52e5d1124335584ab487345e6cd` |
| `Practice/context_2_5d_outputs/patient_aware_last.pth` | 82.4 MB | `9394f562683f493db6208ceb1648db793747803d792364337498003725e9f684` |

### Practice/intensity_robustness_outputs

| `Practice/intensity_robustness_outputs/patient_aware_best.pth` | 82.4 MB | `5f18223c0f618d45d1433dc4a48d950fdea8a039db9ea557629ffd38fe06270f` |
| `Practice/intensity_robustness_outputs/patient_aware_last.pth` | 82.4 MB | `016122d59c487336129b8ada3fa603d89e5a549f1b8999c04a45e0abc86ab12c` |

### Practice/manifest_baseline_smoke_outputs

| `Practice/manifest_baseline_smoke_outputs/baseline_smoke_best.pth` | 82.4 MB | `729dedc90f33ec1d0dfc8c8ce01dcc6efcce6e79ca7aab80afcb01b20e49010e` |
| `Practice/manifest_baseline_smoke_outputs/baseline_smoke_last.pth` | 82.4 MB | `cecff743addf087da11c51445b356b4314c983be689949b4cc5985f59fa8c2ea` |

### Practice/verified_loader_overfit_outputs

| `Practice/verified_loader_overfit_outputs/overfit_last_checkpoint.pth` | 82.4 MB | `b0ebee119b11d58c39397fbcce414a932c368ffce0026aeb1f054e943fa8afe7` |

### models/s9_pilot

| `models/s9_pilot/best_model.pth` | 82.3 MB | `65b024a087ea5c77c27576d5bcd13f9bc456ac67882e71b2938ad70687f9f955` |
| `models/s9_pilot/checkpoint_epoch1.pth` | 82.4 MB | `86ef9c17415ee9d518c675016cc056623106b5d4dfb219893c94cfd3046cdbe7` |
| `models/s9_pilot/checkpoint_epoch2.pth` | 82.4 MB | `bb58e93231862778735911299f750f1f73eb24dd8d8cddc6cf766192c8a3ef2f` |
| `models/s9_pilot/checkpoint_epoch3.pth` | 82.4 MB | `f18da7773155b9e63f53830b6144f0a87c9700247ad34dcd39fd93fa4d75d555` |
| `models/s9_pilot/checkpoint_epoch4.pth` | 82.4 MB | `9765ea41118caf5c0d6d8c6606c837865e413481f602711e6f26ccd31161efdf` |
| `models/s9_pilot/checkpoint_epoch5.pth` | 82.4 MB | `5e3d1d074e65590ad195d0dfcb0ab9a196d766bf874b926d47536264c072c6ea` |
| `models/s9_pilot/final_model.pth` | 82.3 MB | `f2898f50e4392e627b341e0afc0511001fa1efcb87e743ec9de559c0d6514d91` |

### models/s9_pilot_v2

| `models/s9_pilot_v2/best_model.pth` | 82.3 MB | `3949b64d74381e54f823b045d24611c89382c4980eaa16137ff565bf88e408b2` |
| `models/s9_pilot_v2/checkpoint_epoch1.pth` | 82.4 MB | `1e4a55610ce93e6ca3d550c1dcf1725f3ca95a1caf70ea95d34ab61ca2794e14` |
| `models/s9_pilot_v2/checkpoint_epoch2.pth` | 82.4 MB | `c40108a2d01b594e28395e51054c12cbaf493e8bc803df2ebffbbde57a719c22` |
| `models/s9_pilot_v2/checkpoint_epoch3.pth` | 82.4 MB | `40e8c17ed408cc914df95dfaabf2056339dff9e464cfc1aeb6d2e6f990d5e95b` |
| `models/s9_pilot_v2/checkpoint_epoch4.pth` | 82.4 MB | `90354da2f5007f391d91ddf3f33651d09a16df70edec324d1f5f850d4be3cc18` |
| `models/s9_pilot_v2/checkpoint_epoch5.pth` | 82.4 MB | `74b98da8b8600fa246d1c76b353e1f22551a34df09030bb958c31fb814dd6faf` |
| `models/s9_pilot_v2/final_model.pth` | 82.3 MB | `dc16a281608d77aaca4cceedafd4348179d19b79814beb814054da211ce90c5c` |

### models/s9_pilot_v3

| `models/s9_pilot_v3/best_model.pth` | 82.3 MB | `1e65883e0dce614ba7142ef0ac5072833d5498abba657b965fc66babc897faed` |
| `models/s9_pilot_v3/checkpoint_epoch1.pth` | 82.4 MB | `7275fa24a61686016b13d58de834349668ee4ecfeec80785ba6db74dc5405a4e` |
| `models/s9_pilot_v3/checkpoint_epoch2.pth` | 82.4 MB | `538c4e7dceb6d589bd315b43b33dba336be39434b69ed579af6685917bf41be6` |
| `models/s9_pilot_v3/checkpoint_epoch3.pth` | 82.4 MB | `652a90f24c9bb3f0bb045cd3ce213c58853e7e82ea2dbd2002ab54934bf895fb` |
| `models/s9_pilot_v3/checkpoint_epoch4.pth` | 82.4 MB | `440578471fe1b9bfef7afaade994f844dde29956f20095d9d9b543360b7a2faa` |
| `models/s9_pilot_v3/checkpoint_epoch5.pth` | 82.4 MB | `1c00a8f21d1ffe8918412775a0ee1392d8cd77679970c5bd4700055a42e05e63` |
| `models/s9_pilot_v3/final_model.pth` | 82.3 MB | `3f9255b155d82248024278a20c4b5353e293e5314111c7da4765db55292ac520` |

### models/s9_finetune_v4

| `models/s9_finetune_v4/best_model.pth` | 82.3 MB | `2f0d3f3b4696e5184be86136c6acb27a8b0412e8f3847bd081ff7ad46c089bde` |
| `models/s9_finetune_v4/checkpoint_epoch10.pth` | 82.4 MB | `b8aeb350092c82ecfd21dbeb8192e01d2cf8dfd9abb5c79044aa6b948f2ae055` |
| `models/s9_finetune_v4/checkpoint_epoch12.pth` | 82.4 MB | `85fb77235399c5d838b4e22ed9a7892db965df0299b3d4978f87bb26bda6cfcc` |
| `models/s9_finetune_v4/checkpoint_epoch14.pth` | 82.4 MB | `56e03b4b560c0da2fa7bead71a7faf21b6d488b64fa1079da6246e568ef5c43f` |
| `models/s9_finetune_v4/checkpoint_epoch16.pth` | 82.4 MB | `65a7ac5ecf34bb67d14afd441534001e48993281fd8f50b541bd3f66423af3b8` |
| `models/s9_finetune_v4/checkpoint_epoch18.pth` | 82.4 MB | `992b866085076008934d1b4f8432fd742f54d279da5d1651f3d7b0c7bb99d67e` |
| `models/s9_finetune_v4/checkpoint_epoch2.pth` | 82.4 MB | `2aee9695e49b6ddd971aa7736507296a69bf42dd2abeb8b99f4c83375ebcd5bd` |
| `models/s9_finetune_v4/checkpoint_epoch20.pth` | 82.4 MB | `9f778d911af2d00df7e224a7438f95a7ed642e2c12e28c804a5abac1e5b5fc44` |
| `models/s9_finetune_v4/checkpoint_epoch4.pth` | 82.4 MB | `cbe0748f62b3d5872fbed83080f0f0cbf07757a1f83918f96daec00b17875a0f` |
| `models/s9_finetune_v4/checkpoint_epoch6.pth` | 82.4 MB | `4dfd7fe3a8ab30b969aa364496af762117167941ef23e357928f6148b8a56e91` |
| `models/s9_finetune_v4/checkpoint_epoch8.pth` | 82.4 MB | `ed77a1815887db1023901cad89846ee2ac6882d1d8ca6498b6432f50de73e2cf` |
| `models/s9_finetune_v4/final_model.pth` | 82.3 MB | `0334ff4be390fbb528bce802a2446e4df319ba63ec3112cf6650a18d0646b91e` |

### models/research_validation

| `models/research_validation/baseline_focal_dice/best_checkpoint.pth` | 82.4 MB | `c6fdd480c3c8c09856f4dfaca5aab229a4339e4ac34f9bf671c0d9ec2faa3e97` |
| `models/research_validation/baseline_focal_dice/last_checkpoint.pth` | 82.4 MB | `cc680336572cd8bfd4287f53220f487d1f332f4558da1337797788af346c4c69` |
| `models/research_validation/dry_run/best_checkpoint.pth` | 82.4 MB | `514c28362d4e39ef693ed7032071dee828fe466621d4b1189f9c55dde9a489d8` |
| `models/research_validation/dry_run/last_checkpoint.pth` | 82.4 MB | `28c67268b7ec097ed2afd754be9a166b552a807b9e893a8e16538fca6e6e3735` |

### experiments/sprint1

| `experiments/sprint1/sweep_w2/best_model.pth` | 82.3 MB | `47d3212b8624ce7f6e4f2d7286d08c47704f2550cc24143ffaeba150dad3953c` |
| `experiments/sprint1/sweep_w2/checkpoint_epoch1.pth` | 82.4 MB | `6ad59dea56d3a9247f4063b9c49eaeff87a299815d9cc709ff208b2a1a905539` |
| `experiments/sprint1/sweep_w2/final_model.pth` | 82.3 MB | `8e679dca2aeed712d135932c7927c6da289b9ea5ed702d2700adde2172c38296` |
| `experiments/sprint1/sweep_w3/best_model.pth` | 82.3 MB | `38581c89f9a3280ec1b77baf473c91e0a06a16e96899d58335d89b679a4bc120` |
| `experiments/sprint1/sweep_w3/checkpoint_epoch1.pth` | 82.4 MB | `1d429aa53f02e2186c01cae499b853fde9d415a9d793dc67053bf37acd2af141` |
| `experiments/sprint1/sweep_w3/final_model.pth` | 82.3 MB | `f867cc8310b0e9aca256bd487423986632c859a355cab15e919dbac2c481a162` |
| `experiments/sprint1/sweep_w5/best_model.pth` | 82.3 MB | `21997523d598cb84234becf3116e236cbd9a5dcc3b51bb516b26cd4b8ab498bc` |
| `experiments/sprint1/sweep_w5/checkpoint_epoch1.pth` | 82.4 MB | `b97cadec11f9ef12fed12d37ea2c46fa6cea8112b1a973d856f03c4e799c3c18` |
| `experiments/sprint1/sweep_w5/final_model.pth` | 82.3 MB | `9fbcc72c5690e3dd0e31760888c137ff78891c49feaa761bedd91a4fc28dd371` |

### mark 1/mark_3_outputs

| `mark 1/mark_3_outputs/broad_1ch_overfit.pth` | 27.6 MB | `33bec77462b54ac33239bb3fcccf53225aa69bfaf864d119c837b1784bdb0406` |
| `mark 1/mark_3_outputs/broad_liver_2ch_overfit.pth` | 27.6 MB | `03ea1ad945a8583612bf5cc9d14a7f8cfe95ef400a10c9b2b205f356bfbc0dc1` |
| `mark 1/mark_3_outputs/broad_liver_narrow_3ch_overfit.pth` | 27.6 MB | `58f7cb43c4de700ea1d1f5332b7238b56aadd78f1e4dc68e0659aa239a191551` |

### mark 1/mark_4_outputs

| `mark 1/mark_4_outputs/mark_4_best.pth` | 82.3 MB | `9b0c7749af66b0fc3808f6757d0d90af01384e06afe02090361b220df48b6e8b` *(still in working tree — backup copy redundant)* |
| `mark 1/mark_4_outputs/mark_4_last.pth` | 82.3 MB | `a48ed7f97e4b1ac409129e7c4f4f283c7f8ffdd8326460a0ce26814a0add1999` |

### mark 1/mark_4c_outputs

| `mark 1/mark_4c_outputs/recall_loss_best.pth` | 27.6 MB | `1ad1090f98be290ee16933674a79df883064eda0210f4c9eb21477338584c580` *(still in working tree — backup copy redundant)* |
| `mark 1/mark_4c_outputs/two_channel_best.pth` | 27.6 MB | `a015574a8dcceb6713ae0a126475d3ce9c5ab926665d5cc87221e1242085d309` *(still in working tree — backup copy redundant)* |

### Evaluation/output

| `Evaluation/output/03_mark_3/data/broad_1ch_overfit.pth` | 27.6 MB | `33bec77462b54ac33239bb3fcccf53225aa69bfaf864d119c837b1784bdb0406` *(still in working tree — backup copy redundant)* |
| `Evaluation/output/03_mark_3/data/broad_liver_2ch_overfit.pth` | 27.6 MB | `03ea1ad945a8583612bf5cc9d14a7f8cfe95ef400a10c9b2b205f356bfbc0dc1` *(still in working tree — backup copy redundant)* |
| `Evaluation/output/03_mark_3/data/broad_liver_narrow_3ch_overfit.pth` | 27.6 MB | `58f7cb43c4de700ea1d1f5332b7238b56aadd78f1e4dc68e0659aa239a191551` *(still in working tree — backup copy redundant)* |
| `Evaluation/output/06_mark_4c/data/recall_loss_best.pth` | 27.6 MB | `1ad1090f98be290ee16933674a79df883064eda0210f4c9eb21477338584c580` *(still in working tree — backup copy redundant)* |
| `Evaluation/output/06_mark_4c/data/two_channel_best.pth` | 27.6 MB | `a015574a8dcceb6713ae0a126475d3ce9c5ab926665d5cc87221e1242085d309` *(still in working tree — backup copy redundant)* |

### Evaluation/mark_1_to_4e_outputs (legacy mirror, folder deleted; working copy is Evaluation/output)

Legacy mirror removed on 12 Aug 2026; canonical copies live in `Evaluation/output/03_mark_3/data` and `Evaluation/output/06_mark_4c/data`.
| `Evaluation/mark_1_to_4e_outputs/mark_3_outputs/broad_1ch_overfit.pth` | 27.6 MB | `33bec77462b54ac33239bb3fcccf53225aa69bfaf864d119c837b1784bdb0406` |
| `Evaluation/mark_1_to_4e_outputs/mark_3_outputs/broad_liver_2ch_overfit.pth` | 27.6 MB | `03ea1ad945a8583612bf5cc9d14a7f8cfe95ef400a10c9b2b205f356bfbc0dc1` |
| `Evaluation/mark_1_to_4e_outputs/mark_3_outputs/broad_liver_narrow_3ch_overfit.pth` | 27.6 MB | `58f7cb43c4de700ea1d1f5332b7238b56aadd78f1e4dc68e0659aa239a191551` |
| `Evaluation/mark_1_to_4e_outputs/mark_4c_outputs/recall_loss_best.pth` | 27.6 MB | `1ad1090f98be290ee16933674a79df883064eda0210f4c9eb21477338584c580` |
| `Evaluation/mark_1_to_4e_outputs/mark_4c_outputs/two_channel_best.pth` | 27.6 MB | `a015574a8dcceb6713ae0a126475d3ce9c5ab926665d5cc87221e1242085d309` |

## 2. Probability caches — `07_caches_npz_all.zip` (was 3,665 MB / 216 files)

All `.npz` are per-slice forward-pass probability caches, regenerated automatically by their notebooks when missing (see `docs/REPRODUCTION_AND_REGENERATION.md` §3.1). None remain in the working tree. Only path + size + SHA-256 are retained.

| File | Size | SHA-256 |
|---|---|---|
| `Evaluation/mark_1_to_4e_outputs/mark_1_outputs/probability_cache/volume_104.npz` | 87.5 MB | `a3b8462c90322b48e845047fda77c72624b02e101e0bc6e12ea3deed39a86bdd` |
| `Evaluation/mark_1_to_4e_outputs/mark_1_outputs/probability_cache/volume_105.npz` | 110.6 MB | `6871d2c07d89e877a814b192dc68c5621542dc1adb4e8219d18b4907411fef47` |
| `Evaluation/mark_1_to_4e_outputs/mark_1_outputs/probability_cache/volume_106.npz` | 85.1 MB | `28064c9266717a480e75edc7bd416990ceabe019e9d96f1354b1584b5d91485e` |
| `Evaluation/mark_1_to_4e_outputs/mark_1_outputs/probability_cache/volume_107.npz` | 85.5 MB | `5d44fe56fe99444e0ce7ff1983beda6166140054f073b5410a9d6e6d38439cb0` |
| `Evaluation/mark_1_to_4e_outputs/mark_1_outputs/probability_cache/volume_108.npz` | 95.8 MB | `ddee06612828c65f6ae64ebe8bb90046d28c98220a406169beafb68e95977112` |
| `Evaluation/mark_1_to_4e_outputs/mark_1_outputs/probability_cache/volume_109.npz` | 84.2 MB | `dd214c0906f6d018eeb27ceea266dee20ad5c8638832a7d7259f7bc8417f472e` |
| `Evaluation/mark_1_to_4e_outputs/mark_1_outputs/probability_cache/volume_110.npz` | 90.1 MB | `c6772e5518a6c49a75efbc4e8a45db2fe311752fd964dea22937fbde2c29e328` |
| `Evaluation/mark_1_to_4e_outputs/mark_1_outputs/probability_cache/volume_111.npz` | 84.7 MB | `f07ff0afca7c43ddf3433fba8e6899ed422b4cad60fa0a84f06f00da54cd00d0` |
| `Evaluation/mark_1_to_4e_outputs/mark_1_outputs/probability_cache/volume_112.npz` | 83.8 MB | `053b44acedb5efdae200f1090e414a1096909d38ef21c50676564138a04f0b7f` |
| `Evaluation/mark_1_to_4e_outputs/mark_1_outputs/probability_cache/volume_113.npz` | 93.1 MB | `9b0d399224f0fee52244f78a00b4dc01e88fae37b0c2288ad9e3e989c3bdfdb2` |
| `Evaluation/mark_1_to_4e_outputs/mark_1_outputs/probability_cache/volume_114.npz` | 94.1 MB | `128f2f2fc9377c53d534108af9e8552d0ba311e2f237f0cb5852c3e1ca55c052` |
| `Evaluation/mark_1_to_4e_outputs/mark_1_outputs/probability_cache/volume_115.npz` | 95.1 MB | `53a5c6973e2633f0543d2c8870e9ec466899c8e80baa2945d22e40ab523eeb65` |
| `Evaluation/mark_1_to_4e_outputs/mark_1_outputs/probability_cache/volume_116.npz` | 101.6 MB | `1a6ee1022741bb0c3e193a5950c676b1c2af253eab122cea2db4118a346a46ef` |
| `Evaluation/mark_1_to_4e_outputs/mark_4b_outputs/probability_cache/volume_104.npz` | 0.2 MB | `0bf7e690f06bc402776afe0a7d37c8accef8614c194ebb0fa68487e9a080f317` |
| `Evaluation/mark_1_to_4e_outputs/mark_4b_outputs/probability_cache/volume_105.npz` | 0.2 MB | `73a9fc474d78e3bc29c9da3653b9977e01628cd79b370004d2c9358487e682d8` |
| `Evaluation/mark_1_to_4e_outputs/mark_4b_outputs/probability_cache/volume_106.npz` | 0.2 MB | `9f6a8748c39bd70f05d03a3aacde137717034dc28ba2751bf5b47931b5e0d0b9` |
| `Evaluation/mark_1_to_4e_outputs/mark_4b_outputs/probability_cache/volume_107.npz` | 0.2 MB | `2e102011185b7ae0d8bb6b20c4c8da8e00e4d7dff08197ed81b5005a80285013` |
| `Evaluation/mark_1_to_4e_outputs/mark_4b_outputs/probability_cache/volume_108.npz` | 1.6 MB | `42f16cb91c2f76c96310292c556ec9551d4b33357e900544a6cd45f4e9b7fd0b` |
| `Evaluation/mark_1_to_4e_outputs/mark_4b_outputs/probability_cache/volume_109.npz` | 0.3 MB | `e61dbb70ce7fe9eba1033762cd29dc63c0cddaf62f045c069b3e5cb3c9630327` |
| `Evaluation/mark_1_to_4e_outputs/mark_4b_outputs/probability_cache/volume_110.npz` | 0.3 MB | `d0f39bcb700e0d9f419d9baf802d2d8143ea485e85895375cbd11ba94fa62013` |
| `Evaluation/mark_1_to_4e_outputs/mark_4b_outputs/probability_cache/volume_111.npz` | 0.2 MB | `2e7b2b60446ba4347f4022c2ccea0fcf26f2451d91ecdd4db1b170ddf96b29c6` |
| `Evaluation/mark_1_to_4e_outputs/mark_4b_outputs/probability_cache/volume_112.npz` | 0.2 MB | `d58f6681a515a9f22ff6bcb07ee320ce09bcb3ca248b212ad17232d6072e0398` |
| `Evaluation/mark_1_to_4e_outputs/mark_4b_outputs/probability_cache/volume_113.npz` | 0.3 MB | `9735926c15c0d207b5543d6ab3d3780c4bca405d3d60995c4e68d9ade781b940` |
| `Evaluation/mark_1_to_4e_outputs/mark_4b_outputs/probability_cache/volume_114.npz` | 0.2 MB | `2f3949daa3117fbc9a437a7e5666514fab5af43f14a9e29ea717039901daef2b` |
| `Evaluation/mark_1_to_4e_outputs/mark_4b_outputs/probability_cache/volume_115.npz` | 0.2 MB | `5afdba49fd51be1ae2292f053a08f24e2dfbdc6099b7822f2b79956be8986d4c` |
| `Evaluation/mark_1_to_4e_outputs/mark_4b_outputs/probability_cache/volume_116.npz` | 0.2 MB | `b677dd569c795a9d2daad706d6d7e610d1749ef47830bc25639d53df217f5c5c` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/control/volume_104.npz` | 0.2 MB | `a241de8afcd6086e7f75cda4b8ec354f6f7894f10e1426e6df062bb6bae5b2dd` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/control/volume_105.npz` | 0.2 MB | `260eb563ef9c1f5296e29bb494c9817b923f068268a2d96dd6dd6c2f5b5ee788` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/control/volume_106.npz` | 0.2 MB | `6afc5c3f11ff8440a9549eb5fe64d41b70991ec88285d76020a4fb0687230ebd` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/control/volume_107.npz` | 0.2 MB | `9d950c866b0048f468d401a441244bc0a9f0a56b166d9bb694af70d08779b6a6` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/control/volume_108.npz` | 1.6 MB | `7ca9525b413f65acab7b419e79ebadc9b8cd769ad6a66d9961b6adfaf7dff9ef` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/control/volume_109.npz` | 0.3 MB | `100225e61a1bdb7a6948a596596a334e28e470c0c1f167253bfb84f88983518e` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/control/volume_110.npz` | 0.3 MB | `cea94a1022fd626720e210ee2c3275f52a602bf45ba6fb988343bee0ef2067d7` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/control/volume_111.npz` | 0.2 MB | `61de7873af5957caf8aaece22c60f8334c3a30b585b7651f0eadb116c6cc9a92` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/control/volume_112.npz` | 0.2 MB | `f1db962d6efd7d24f8be20f4a7b53bbfc21eae3c1db722335414f6ac28d676ec` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/control/volume_113.npz` | 0.3 MB | `678cb84198d46afe299dc5812e4070e8256c716b89ee79226f545d6858979e09` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/control/volume_114.npz` | 0.2 MB | `98ab0c3978c07d62bbabd68db412115b59032df3627b5331ed46b82620e0a132` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/control/volume_115.npz` | 0.2 MB | `5e3f5f77fe6882d7da7040fe74576dd9213f66a32d8e7ae1726fb8e749c49a84` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/control/volume_116.npz` | 0.2 MB | `379a64ef6e644e51d1459b010e456088ec8daea72fffbd170f4dc35190a3c223` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/recall_loss/volume_104.npz` | 0.2 MB | `334fc9460ff8b98f2b4e8b685e53babd3752a092b28a82bc8050414e3cb59277` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/recall_loss/volume_105.npz` | 0.2 MB | `1273e2b6d517a0e1e5b9ec92929132dc8acb61f8148c553ad396a6642b3e5e9f` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/recall_loss/volume_106.npz` | 0.2 MB | `499eb18d3aebb3986187fc7369bb80a3ef20b804cceb417b2fe64b02a215da50` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/recall_loss/volume_107.npz` | 0.2 MB | `f4cb0708402192c908c24da53bb0bc2d7461c542c344cc9077dc556d34616980` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/recall_loss/volume_108.npz` | 1.2 MB | `cec4eb19eeb64a4c9c7d3fd46f4eb7194837183d950f7918faf6a78e8fc4625c` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/recall_loss/volume_109.npz` | 0.3 MB | `e913d65fa756f459a8b41cfffa66c479755a2c4257c04e309fe8c443572d5ed1` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/recall_loss/volume_110.npz` | 0.3 MB | `16b1d83b9ca5e6fa715966f05efe9cf14d57dad7e8bd94ea1c64c9076a33e240` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/recall_loss/volume_111.npz` | 0.2 MB | `f567d8a9b1c5b274a550b7493c428c3612a38fa64329d219de2c0bf9f3015b5d` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/recall_loss/volume_112.npz` | 0.2 MB | `712561eb6e962fe1eddf7822284c72e70ead12a879b8760af95781e44c004409` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/recall_loss/volume_113.npz` | 0.3 MB | `8daeeba82c7c6662a96491fca6cf7ca7fbbfbc07de7fb3a3be710d7478a7a709` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/recall_loss/volume_114.npz` | 0.2 MB | `5a2c27bc962b04d945109816b244131f584974ef2bbe8cd9ab44e446353cb2f6` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/recall_loss/volume_115.npz` | 0.2 MB | `a1e8897f6c1eb5bbd1f7c46ac608bcfb6291e8f0de4d94504b850bcf7a9b2b57` |
| `Evaluation/mark_1_to_4e_outputs/mark_4d_outputs/probability_cache/recall_loss/volume_116.npz` | 0.2 MB | `3fa069087fc62d2c641f51faab33ce842c15a5a689023a2d481f08a88eb086c5` |
| `Evaluation/output/01_mark_1/caches/volume_104.npz` | 87.5 MB | `a3b8462c90322b48e845047fda77c72624b02e101e0bc6e12ea3deed39a86bdd` |
| `Evaluation/output/01_mark_1/caches/volume_105.npz` | 110.6 MB | `6871d2c07d89e877a814b192dc68c5621542dc1adb4e8219d18b4907411fef47` |
| `Evaluation/output/01_mark_1/caches/volume_106.npz` | 85.1 MB | `28064c9266717a480e75edc7bd416990ceabe019e9d96f1354b1584b5d91485e` |
| `Evaluation/output/01_mark_1/caches/volume_107.npz` | 85.5 MB | `5d44fe56fe99444e0ce7ff1983beda6166140054f073b5410a9d6e6d38439cb0` |
| `Evaluation/output/01_mark_1/caches/volume_108.npz` | 95.8 MB | `ddee06612828c65f6ae64ebe8bb90046d28c98220a406169beafb68e95977112` |
| `Evaluation/output/01_mark_1/caches/volume_109.npz` | 84.2 MB | `dd214c0906f6d018eeb27ceea266dee20ad5c8638832a7d7259f7bc8417f472e` |
| `Evaluation/output/01_mark_1/caches/volume_110.npz` | 90.1 MB | `c6772e5518a6c49a75efbc4e8a45db2fe311752fd964dea22937fbde2c29e328` |
| `Evaluation/output/01_mark_1/caches/volume_111.npz` | 84.7 MB | `f07ff0afca7c43ddf3433fba8e6899ed422b4cad60fa0a84f06f00da54cd00d0` |
| `Evaluation/output/01_mark_1/caches/volume_112.npz` | 83.8 MB | `053b44acedb5efdae200f1090e414a1096909d38ef21c50676564138a04f0b7f` |
| `Evaluation/output/01_mark_1/caches/volume_113.npz` | 93.1 MB | `9b0d399224f0fee52244f78a00b4dc01e88fae37b0c2288ad9e3e989c3bdfdb2` |
| `Evaluation/output/01_mark_1/caches/volume_114.npz` | 94.1 MB | `128f2f2fc9377c53d534108af9e8552d0ba311e2f237f0cb5852c3e1ca55c052` |
| `Evaluation/output/01_mark_1/caches/volume_115.npz` | 95.1 MB | `53a5c6973e2633f0543d2c8870e9ec466899c8e80baa2945d22e40ab523eeb65` |
| `Evaluation/output/01_mark_1/caches/volume_116.npz` | 101.6 MB | `1a6ee1022741bb0c3e193a5950c676b1c2af253eab122cea2db4118a346a46ef` |
| `Evaluation/output/05_mark_4b/caches/volume_104.npz` | 0.2 MB | `0bf7e690f06bc402776afe0a7d37c8accef8614c194ebb0fa68487e9a080f317` |
| `Evaluation/output/05_mark_4b/caches/volume_105.npz` | 0.2 MB | `73a9fc474d78e3bc29c9da3653b9977e01628cd79b370004d2c9358487e682d8` |
| `Evaluation/output/05_mark_4b/caches/volume_106.npz` | 0.2 MB | `9f6a8748c39bd70f05d03a3aacde137717034dc28ba2751bf5b47931b5e0d0b9` |
| `Evaluation/output/05_mark_4b/caches/volume_107.npz` | 0.2 MB | `2e102011185b7ae0d8bb6b20c4c8da8e00e4d7dff08197ed81b5005a80285013` |
| `Evaluation/output/05_mark_4b/caches/volume_108.npz` | 1.6 MB | `42f16cb91c2f76c96310292c556ec9551d4b33357e900544a6cd45f4e9b7fd0b` |
| `Evaluation/output/05_mark_4b/caches/volume_109.npz` | 0.3 MB | `e61dbb70ce7fe9eba1033762cd29dc63c0cddaf62f045c069b3e5cb3c9630327` |
| `Evaluation/output/05_mark_4b/caches/volume_110.npz` | 0.3 MB | `d0f39bcb700e0d9f419d9baf802d2d8143ea485e85895375cbd11ba94fa62013` |
| `Evaluation/output/05_mark_4b/caches/volume_111.npz` | 0.2 MB | `2e7b2b60446ba4347f4022c2ccea0fcf26f2451d91ecdd4db1b170ddf96b29c6` |
| `Evaluation/output/05_mark_4b/caches/volume_112.npz` | 0.2 MB | `d58f6681a515a9f22ff6bcb07ee320ce09bcb3ca248b212ad17232d6072e0398` |
| `Evaluation/output/05_mark_4b/caches/volume_113.npz` | 0.3 MB | `9735926c15c0d207b5543d6ab3d3780c4bca405d3d60995c4e68d9ade781b940` |
| `Evaluation/output/05_mark_4b/caches/volume_114.npz` | 0.2 MB | `2f3949daa3117fbc9a437a7e5666514fab5af43f14a9e29ea717039901daef2b` |
| `Evaluation/output/05_mark_4b/caches/volume_115.npz` | 0.2 MB | `5afdba49fd51be1ae2292f053a08f24e2dfbdc6099b7822f2b79956be8986d4c` |
| `Evaluation/output/05_mark_4b/caches/volume_116.npz` | 0.2 MB | `b677dd569c795a9d2daad706d6d7e610d1749ef47830bc25639d53df217f5c5c` |
| `Evaluation/output/07_mark_4d/caches/control/volume_104.npz` | 0.2 MB | `a241de8afcd6086e7f75cda4b8ec354f6f7894f10e1426e6df062bb6bae5b2dd` |
| `Evaluation/output/07_mark_4d/caches/control/volume_105.npz` | 0.2 MB | `260eb563ef9c1f5296e29bb494c9817b923f068268a2d96dd6dd6c2f5b5ee788` |
| `Evaluation/output/07_mark_4d/caches/control/volume_106.npz` | 0.2 MB | `6afc5c3f11ff8440a9549eb5fe64d41b70991ec88285d76020a4fb0687230ebd` |
| `Evaluation/output/07_mark_4d/caches/control/volume_107.npz` | 0.2 MB | `9d950c866b0048f468d401a441244bc0a9f0a56b166d9bb694af70d08779b6a6` |
| `Evaluation/output/07_mark_4d/caches/control/volume_108.npz` | 1.6 MB | `7ca9525b413f65acab7b419e79ebadc9b8cd769ad6a66d9961b6adfaf7dff9ef` |
| `Evaluation/output/07_mark_4d/caches/control/volume_109.npz` | 0.3 MB | `100225e61a1bdb7a6948a596596a334e28e470c0c1f167253bfb84f88983518e` |
| `Evaluation/output/07_mark_4d/caches/control/volume_110.npz` | 0.3 MB | `cea94a1022fd626720e210ee2c3275f52a602bf45ba6fb988343bee0ef2067d7` |
| `Evaluation/output/07_mark_4d/caches/control/volume_111.npz` | 0.2 MB | `61de7873af5957caf8aaece22c60f8334c3a30b585b7651f0eadb116c6cc9a92` |
| `Evaluation/output/07_mark_4d/caches/control/volume_112.npz` | 0.2 MB | `f1db962d6efd7d24f8be20f4a7b53bbfc21eae3c1db722335414f6ac28d676ec` |
| `Evaluation/output/07_mark_4d/caches/control/volume_113.npz` | 0.3 MB | `678cb84198d46afe299dc5812e4070e8256c716b89ee79226f545d6858979e09` |
| `Evaluation/output/07_mark_4d/caches/control/volume_114.npz` | 0.2 MB | `98ab0c3978c07d62bbabd68db412115b59032df3627b5331ed46b82620e0a132` |
| `Evaluation/output/07_mark_4d/caches/control/volume_115.npz` | 0.2 MB | `5e3f5f77fe6882d7da7040fe74576dd9213f66a32d8e7ae1726fb8e749c49a84` |
| `Evaluation/output/07_mark_4d/caches/control/volume_116.npz` | 0.2 MB | `379a64ef6e644e51d1459b010e456088ec8daea72fffbd170f4dc35190a3c223` |
| `Evaluation/output/07_mark_4d/caches/recall_loss/volume_104.npz` | 0.2 MB | `334fc9460ff8b98f2b4e8b685e53babd3752a092b28a82bc8050414e3cb59277` |
| `Evaluation/output/07_mark_4d/caches/recall_loss/volume_105.npz` | 0.2 MB | `1273e2b6d517a0e1e5b9ec92929132dc8acb61f8148c553ad396a6642b3e5e9f` |
| `Evaluation/output/07_mark_4d/caches/recall_loss/volume_106.npz` | 0.2 MB | `499eb18d3aebb3986187fc7369bb80a3ef20b804cceb417b2fe64b02a215da50` |
| `Evaluation/output/07_mark_4d/caches/recall_loss/volume_107.npz` | 0.2 MB | `f4cb0708402192c908c24da53bb0bc2d7461c542c344cc9077dc556d34616980` |
| `Evaluation/output/07_mark_4d/caches/recall_loss/volume_108.npz` | 1.2 MB | `cec4eb19eeb64a4c9c7d3fd46f4eb7194837183d950f7918faf6a78e8fc4625c` |
| `Evaluation/output/07_mark_4d/caches/recall_loss/volume_109.npz` | 0.3 MB | `e913d65fa756f459a8b41cfffa66c479755a2c4257c04e309fe8c443572d5ed1` |
| `Evaluation/output/07_mark_4d/caches/recall_loss/volume_110.npz` | 0.3 MB | `16b1d83b9ca5e6fa715966f05efe9cf14d57dad7e8bd94ea1c64c9076a33e240` |
| `Evaluation/output/07_mark_4d/caches/recall_loss/volume_111.npz` | 0.2 MB | `f567d8a9b1c5b274a550b7493c428c3612a38fa64329d219de2c0bf9f3015b5d` |
| `Evaluation/output/07_mark_4d/caches/recall_loss/volume_112.npz` | 0.2 MB | `712561eb6e962fe1eddf7822284c72e70ead12a879b8760af95781e44c004409` |
| `Evaluation/output/07_mark_4d/caches/recall_loss/volume_113.npz` | 0.3 MB | `8daeeba82c7c6662a96491fca6cf7ca7fbbfbc07de7fb3a3be710d7478a7a709` |
| `Evaluation/output/07_mark_4d/caches/recall_loss/volume_114.npz` | 0.2 MB | `5a2c27bc962b04d945109816b244131f584974ef2bbe8cd9ab44e446353cb2f6` |
| `Evaluation/output/07_mark_4d/caches/recall_loss/volume_115.npz` | 0.2 MB | `a1e8897f6c1eb5bbd1f7c46ac608bcfb6291e8f0de4d94504b850bcf7a9b2b57` |
| `Evaluation/output/07_mark_4d/caches/recall_loss/volume_116.npz` | 0.2 MB | `3fa069087fc62d2c641f51faab33ce842c15a5a689023a2d481f08a88eb086c5` |
| `Practice/validation_3d_postprocessing_outputs/probability_cache/volume_104.npz` | 1.3 MB | `b00477e2a9ca5959cf2f2b82ea86b4e19b27c6ef9799f5d0c3f9c7860e3a824b` |
| `Practice/validation_3d_postprocessing_outputs/probability_cache/volume_105.npz` | 2.1 MB | `b1b457b19cf32b8a21428e1987f9720ae622983f5df644975cc55d5e5b039d9e` |
| `Practice/validation_3d_postprocessing_outputs/probability_cache/volume_106.npz` | 1.5 MB | `ad5f374c1a17836b9aefac4fff048111df5b82d125d9b98c375f14d1f0494416` |
| `Practice/validation_3d_postprocessing_outputs/probability_cache/volume_107.npz` | 1.9 MB | `5b4946997a3a2cb7aec998e7c6f20aa97b9a20f30b51b62fb32b6afb0ba14fc8` |
| `Practice/validation_3d_postprocessing_outputs/probability_cache/volume_108.npz` | 2.3 MB | `a2acd70de44036a306ff49639b6a3b6b036c43634edfa3b561ddf883efdfcb37` |
| `Practice/validation_3d_postprocessing_outputs/probability_cache/volume_109.npz` | 1.9 MB | `2f597e2398427e9daa078243f8db5136919b8ff173a3aad184cb2536a78fd96d` |
| `Practice/validation_3d_postprocessing_outputs/probability_cache/volume_110.npz` | 2.5 MB | `47e1d5838c3ad8168dc6d7c86522edd4d17226dafe659351b4a45c1b1ce31d9c` |
| `Practice/validation_3d_postprocessing_outputs/probability_cache/volume_111.npz` | 1.7 MB | `0b5ba332a46a83664c7d1c48153e4c2d9e57c9751150ada480b62c750d81eee0` |
| `Practice/validation_3d_postprocessing_outputs/probability_cache/volume_112.npz` | 1.7 MB | `8c906774c6f1d5fbffcafbb81a5d68420413679372d7455b0fff67d199156c5b` |
| `Practice/validation_3d_postprocessing_outputs/probability_cache/volume_113.npz` | 1.7 MB | `f95f724d50bd4a68a32ce26145a0c8448f8d9c68dfb2649103c9c428bd50f360` |
| `Practice/validation_3d_postprocessing_outputs/probability_cache/volume_114.npz` | 2.5 MB | `062c663bf0a225429955c93b911432ad69f33911a392eae299e562528bcd2cf6` |
| `Practice/validation_3d_postprocessing_outputs/probability_cache/volume_115.npz` | 1.2 MB | `8013a107f15dc055621d53a188de086079bd8931339b734ed5c4ac980430b2d0` |
| `Practice/validation_3d_postprocessing_outputs/probability_cache/volume_116.npz` | 2.0 MB | `745f2c1ad68112c9a2b9feab676aac27633120955eac6c079588965994823fc4` |
| `mark 1 (part 2)/step_02_fusion_freeze_confirmation/outputs/probability_cache/volume_104.npz` | 0.6 MB | `a5bfe86ce57dac8e574402f131c7b1b4e11b186248fc340d2a36bcdb60328df5` |
| `mark 1 (part 2)/step_02_fusion_freeze_confirmation/outputs/probability_cache/volume_105.npz` | 0.6 MB | `a8252054618b6d6b931d4f9f57a3cef18276701fd068cf89c0e30ddb39d70120` |
| `mark 1 (part 2)/step_02_fusion_freeze_confirmation/outputs/probability_cache/volume_106.npz` | 0.5 MB | `45b9c960bb73f8da66540c07ec24477395d3543007e70f0ffb59f511459de57c` |
| `mark 1 (part 2)/step_02_fusion_freeze_confirmation/outputs/probability_cache/volume_107.npz` | 0.6 MB | `f36de352e90df7d9b4ec5b408251f6d42aaf16223492577f659e41b91930282a` |
| `mark 1 (part 2)/step_02_fusion_freeze_confirmation/outputs/probability_cache/volume_108.npz` | 4.2 MB | `d3242a266e7d05f05beab517e059f925920ae20865dac5dbbd1af71e00c4f38a` |
| `mark 1 (part 2)/step_02_fusion_freeze_confirmation/outputs/probability_cache/volume_109.npz` | 0.8 MB | `b32dbc9ec59a114f0bedf955dfe1bc38aef2b713cb6b04c836275cccf155422a` |
| `mark 1 (part 2)/step_02_fusion_freeze_confirmation/outputs/probability_cache/volume_110.npz` | 0.8 MB | `0442bba418af643cb67b4d2c68e95a92e80f22d0b4edea3faa22567b9da270c7` |
| `mark 1 (part 2)/step_02_fusion_freeze_confirmation/outputs/probability_cache/volume_111.npz` | 0.6 MB | `1c11f9ec9bc518d583a1682b706ea2916cb070de5907f91dc300f84298359e67` |
| `mark 1 (part 2)/step_02_fusion_freeze_confirmation/outputs/probability_cache/volume_112.npz` | 0.5 MB | `a0f302c2aa38e1a99f03987034f109f5838d97545eac765f530cc16a15955193` |
| `mark 1 (part 2)/step_02_fusion_freeze_confirmation/outputs/probability_cache/volume_113.npz` | 0.9 MB | `d4eb834ee854df9c5a0cd4b19bbbce03e5b5c13ab90bca91672aaeb6e30faedb` |
| `mark 1 (part 2)/step_02_fusion_freeze_confirmation/outputs/probability_cache/volume_114.npz` | 0.5 MB | `236ed468a4f2120bf350e54bdc3f3994590354863baaabafd6cbc238c0e75322` |
| `mark 1 (part 2)/step_02_fusion_freeze_confirmation/outputs/probability_cache/volume_115.npz` | 0.7 MB | `5b70dabb33ffb0b9fe7de741328ca4dde20843474a34cb4f217ca6765f47c271` |
| `mark 1 (part 2)/step_02_fusion_freeze_confirmation/outputs/probability_cache/volume_116.npz` | 0.6 MB | `ef52a775f5d8a6c8f8a25d011283d3b54fc2052e252c3cdc225e020b11771185` |
| `mark 1 (part 2)/step_04_one_time_locked_test_evaluation_after_explicit_authorization/outputs/probability_cache/volume_117.npz` | 2.9 MB | `b98a20cedeb5b0ad9d44b97eebff5445b8b552ba86b6f5c9dbb0235188f7d740` |
| `mark 1 (part 2)/step_04_one_time_locked_test_evaluation_after_explicit_authorization/outputs/probability_cache/volume_118.npz` | 0.8 MB | `a56f71eb04f1338630660805d46d63908e701aefefcb0ad57cdf0a01766e6872` |
| `mark 1 (part 2)/step_04_one_time_locked_test_evaluation_after_explicit_authorization/outputs/probability_cache/volume_119.npz` | 0.3 MB | `a20e23347b8779e09a6748702a3c20571b8152b80df99ddd4847c145ad9f6b41` |
| `mark 1 (part 2)/step_04_one_time_locked_test_evaluation_after_explicit_authorization/outputs/probability_cache/volume_120.npz` | 0.4 MB | `e0285336239532b18cf44a83eab1024301f2a111b11d699786ca93193fa4071f` |
| `mark 1 (part 2)/step_04_one_time_locked_test_evaluation_after_explicit_authorization/outputs/probability_cache/volume_121.npz` | 0.3 MB | `39e6fcc2ad78f6369ee75a8b04829bbebadfaad0cae6a5facdd499fe8d5f5169` |
| `mark 1 (part 2)/step_04_one_time_locked_test_evaluation_after_explicit_authorization/outputs/probability_cache/volume_122.npz` | 0.4 MB | `11e5939e6dd80616c990b605b393308956c64bf39cfc773a189d801e06963e18` |
| `mark 1 (part 2)/step_04_one_time_locked_test_evaluation_after_explicit_authorization/outputs/probability_cache/volume_123.npz` | 0.7 MB | `ff6cbb5a97fa7d1d762a9bc40c0a19f9334869c06dc2f935be4339489b1a3bc4` |
| `mark 1 (part 2)/step_04_one_time_locked_test_evaluation_after_explicit_authorization/outputs/probability_cache/volume_124.npz` | 0.4 MB | `bdd6739285044f88807af21b46606bbf9fe4017177a409e4529e7d6f1432f10e` |
| `mark 1 (part 2)/step_04_one_time_locked_test_evaluation_after_explicit_authorization/outputs/probability_cache/volume_125.npz` | 0.3 MB | `00d7a83bed976246bb3308a59f8cfce85634a72336833087806ed86577104545` |
| `mark 1 (part 2)/step_04_one_time_locked_test_evaluation_after_explicit_authorization/outputs/probability_cache/volume_126.npz` | 0.3 MB | `c307665b77dd2e22143e1af9442ac22a3a6a2a5fe6433a4ad58b644a39e16143` |
| `mark 1 (part 2)/step_04_one_time_locked_test_evaluation_after_explicit_authorization/outputs/probability_cache/volume_127.npz` | 0.6 MB | `6b74bfb143797376772eb2cc9a55261632a3bcc9e071bc275539dc4c23dc8964` |
| `mark 1 (part 2)/step_04_one_time_locked_test_evaluation_after_explicit_authorization/outputs/probability_cache/volume_128.npz` | 1.6 MB | `e63a717553959914c6577205009cea34fe5fa125f5e8e70b3e81e9b526a91756` |
| `mark 1 (part 2)/step_04_one_time_locked_test_evaluation_after_explicit_authorization/outputs/probability_cache/volume_129.npz` | 2.8 MB | `466e721b49666eab93ba667adad6941a051eb6570520d03c324473171d657643` |
| `mark 1 (part 2)/step_04_one_time_locked_test_evaluation_after_explicit_authorization/outputs/probability_cache/volume_130.npz` | 2.2 MB | `eed182fc0e58ab15b3f871c5dbe38f265de7a77155c23a36ce439731f348782e` |
| `mark 1 (part 2)/step_16_one_time_external_evaluation_after_explicit_authorization/outputs/probability_cache/ircadb_01.npz` | 0.5 MB | `dfb626800f2d33dee566bb86823989bb78fedbf14c6a570c5e279e904b62a58c` |
| `mark 1 (part 2)/step_16_one_time_external_evaluation_after_explicit_authorization/outputs/probability_cache/ircadb_02.npz` | 0.2 MB | `5f69ea20a34401fe807c8f15f167268ea2542d4688122ae41d41332b7fa37fd1` |
| `mark 1 (part 2)/step_16_one_time_external_evaluation_after_explicit_authorization/outputs/probability_cache/ircadb_03.npz` | 0.3 MB | `59998f06ce9971704ffbd4eb98af2ad8d52617d3d76f1d0c23636bf895ea2050` |
| `mark 1 (part 2)/step_16_one_time_external_evaluation_after_explicit_authorization/outputs/probability_cache/ircadb_04.npz` | 0.1 MB | `10454c72d41f3ab9979a0e9fb0e3d1503f9fdb762a6f33d8c3f8d8b2e1dc82e8` |
| `mark 1 (part 2)/step_16_one_time_external_evaluation_after_explicit_authorization/outputs/probability_cache/ircadb_05.npz` | 0.1 MB | `ca7267cfebc8c829076bb080e0a67e23ff54c9794eacc24e29010c112a6673eb` |
| `mark 1 (part 2)/step_16_one_time_external_evaluation_after_explicit_authorization/outputs/probability_cache/ircadb_06.npz` | 0.9 MB | `f920c221efc3ee5ed4feff1d19276e8a5fbd4125d06aff987be75d51c1881997` |
| `mark 1 (part 2)/step_16_one_time_external_evaluation_after_explicit_authorization/outputs/probability_cache/ircadb_07.npz` | 0.2 MB | `301de9911afb11cf8227b91542ef0afab64b1962c7e7b64dca55bc062ad24b70` |
| `mark 1 (part 2)/step_16_one_time_external_evaluation_after_explicit_authorization/outputs/probability_cache/ircadb_08.npz` | 0.2 MB | `5a1a29df73b463ae0d583b3dce95893931d8f28563cacbd4934d3974cad4d360` |
| `mark 1 (part 2)/step_16_one_time_external_evaluation_after_explicit_authorization/outputs/probability_cache/ircadb_09.npz` | 0.2 MB | `e3058997a6f97ab5ab74132e8b4e2ea6c5d63336b6e4897a14bd3c13597e8e1c` |
| `mark 1 (part 2)/step_16_one_time_external_evaluation_after_explicit_authorization/outputs/probability_cache/ircadb_10.npz` | 0.2 MB | `0f295bd2e9dd993057ce7f495c87d3392b76e5bc8e20b457352e3c3a04fc18dd` |
| `mark 1 (part 2)/step_16_one_time_external_evaluation_after_explicit_authorization/outputs/probability_cache/ircadb_11.npz` | 0.1 MB | `29d3295939027d75197eee17067f2b5d975a6d885cd8009d562cc0c97f722308` |
| `mark 1 (part 2)/step_16_one_time_external_evaluation_after_explicit_authorization/outputs/probability_cache/ircadb_12.npz` | 0.6 MB | `8806bbc9aa1fe110ed1763454292917a270a6c4c8e97b3b3f9a7ed6780acdc4d` |
| `mark 1 (part 2)/step_16_one_time_external_evaluation_after_explicit_authorization/outputs/probability_cache/ircadb_13.npz` | 0.4 MB | `393d34b0de5b128eb842ba1814cb201d290cf45152451c40415ec96b018fdf87` |
| `mark 1 (part 2)/step_16_one_time_external_evaluation_after_explicit_authorization/outputs/probability_cache/ircadb_14.npz` | 0.2 MB | `4f882a54a2bc0c662bf67559ca9a95b21d2d39068a0835cd22f427750422ce59` |
| `mark 1 (part 2)/step_16_one_time_external_evaluation_after_explicit_authorization/outputs/probability_cache/ircadb_15.npz` | 0.1 MB | `18f3d1b0f9038c19137c474b6f543dec21292afd6d2de3157454aa441db0a992` |
| `mark 1 (part 2)/step_16_one_time_external_evaluation_after_explicit_authorization/outputs/probability_cache/ircadb_16.npz` | 0.2 MB | `05e73df7b6e5ec2fa96eb16d0485c97ca79e4327a7ce65df404656ca35eb305f` |
| `mark 1 (part 2)/step_16_one_time_external_evaluation_after_explicit_authorization/outputs/probability_cache/ircadb_17.npz` | 0.4 MB | `ed33abd64159afed5b3968d7c33ab59884004faa4c57540d067fb6333ac51aeb` |
| `mark 1 (part 2)/step_16_one_time_external_evaluation_after_explicit_authorization/outputs/probability_cache/ircadb_18.npz` | 0.1 MB | `bd57f97131ed3ff43300575e0b189a27da9789c1915518a915a26fce5eef60ee` |
| `mark 1 (part 2)/step_16_one_time_external_evaluation_after_explicit_authorization/outputs/probability_cache/ircadb_19.npz` | 0.2 MB | `17829c3a78037b15d3946257caefff49397259e24cc7ef87e418b30eb8f5e635` |
| `mark 1 (part 2)/step_16_one_time_external_evaluation_after_explicit_authorization/outputs/probability_cache/ircadb_20.npz` | 0.1 MB | `d9c11d4a1a036f1d2c6a681907b773f5eeb571e8df32b6ddc651fb6f06ccce90` |
| `mark 1/mark_1_outputs/probability_cache/volume_104.npz` | 87.5 MB | `a3b8462c90322b48e845047fda77c72624b02e101e0bc6e12ea3deed39a86bdd` |
| `mark 1/mark_1_outputs/probability_cache/volume_105.npz` | 110.6 MB | `6871d2c07d89e877a814b192dc68c5621542dc1adb4e8219d18b4907411fef47` |
| `mark 1/mark_1_outputs/probability_cache/volume_106.npz` | 85.1 MB | `28064c9266717a480e75edc7bd416990ceabe019e9d96f1354b1584b5d91485e` |
| `mark 1/mark_1_outputs/probability_cache/volume_107.npz` | 85.5 MB | `5d44fe56fe99444e0ce7ff1983beda6166140054f073b5410a9d6e6d38439cb0` |
| `mark 1/mark_1_outputs/probability_cache/volume_108.npz` | 95.8 MB | `ddee06612828c65f6ae64ebe8bb90046d28c98220a406169beafb68e95977112` |
| `mark 1/mark_1_outputs/probability_cache/volume_109.npz` | 84.2 MB | `dd214c0906f6d018eeb27ceea266dee20ad5c8638832a7d7259f7bc8417f472e` |
| `mark 1/mark_1_outputs/probability_cache/volume_110.npz` | 90.1 MB | `c6772e5518a6c49a75efbc4e8a45db2fe311752fd964dea22937fbde2c29e328` |
| `mark 1/mark_1_outputs/probability_cache/volume_111.npz` | 84.7 MB | `f07ff0afca7c43ddf3433fba8e6899ed422b4cad60fa0a84f06f00da54cd00d0` |
| `mark 1/mark_1_outputs/probability_cache/volume_112.npz` | 83.8 MB | `053b44acedb5efdae200f1090e414a1096909d38ef21c50676564138a04f0b7f` |
| `mark 1/mark_1_outputs/probability_cache/volume_113.npz` | 93.1 MB | `9b0d399224f0fee52244f78a00b4dc01e88fae37b0c2288ad9e3e989c3bdfdb2` |
| `mark 1/mark_1_outputs/probability_cache/volume_114.npz` | 94.1 MB | `128f2f2fc9377c53d534108af9e8552d0ba311e2f237f0cb5852c3e1ca55c052` |
| `mark 1/mark_1_outputs/probability_cache/volume_115.npz` | 95.1 MB | `53a5c6973e2633f0543d2c8870e9ec466899c8e80baa2945d22e40ab523eeb65` |
| `mark 1/mark_1_outputs/probability_cache/volume_116.npz` | 101.6 MB | `1a6ee1022741bb0c3e193a5950c676b1c2af253eab122cea2db4118a346a46ef` |
| `mark 1/mark_4b_outputs/probability_cache/volume_104.npz` | 0.2 MB | `0bf7e690f06bc402776afe0a7d37c8accef8614c194ebb0fa68487e9a080f317` |
| `mark 1/mark_4b_outputs/probability_cache/volume_105.npz` | 0.2 MB | `73a9fc474d78e3bc29c9da3653b9977e01628cd79b370004d2c9358487e682d8` |
| `mark 1/mark_4b_outputs/probability_cache/volume_106.npz` | 0.2 MB | `9f6a8748c39bd70f05d03a3aacde137717034dc28ba2751bf5b47931b5e0d0b9` |
| `mark 1/mark_4b_outputs/probability_cache/volume_107.npz` | 0.2 MB | `2e102011185b7ae0d8bb6b20c4c8da8e00e4d7dff08197ed81b5005a80285013` |
| `mark 1/mark_4b_outputs/probability_cache/volume_108.npz` | 1.6 MB | `42f16cb91c2f76c96310292c556ec9551d4b33357e900544a6cd45f4e9b7fd0b` |
| `mark 1/mark_4b_outputs/probability_cache/volume_109.npz` | 0.3 MB | `e61dbb70ce7fe9eba1033762cd29dc63c0cddaf62f045c069b3e5cb3c9630327` |
| `mark 1/mark_4b_outputs/probability_cache/volume_110.npz` | 0.3 MB | `d0f39bcb700e0d9f419d9baf802d2d8143ea485e85895375cbd11ba94fa62013` |
| `mark 1/mark_4b_outputs/probability_cache/volume_111.npz` | 0.2 MB | `2e7b2b60446ba4347f4022c2ccea0fcf26f2451d91ecdd4db1b170ddf96b29c6` |
| `mark 1/mark_4b_outputs/probability_cache/volume_112.npz` | 0.2 MB | `d58f6681a515a9f22ff6bcb07ee320ce09bcb3ca248b212ad17232d6072e0398` |
| `mark 1/mark_4b_outputs/probability_cache/volume_113.npz` | 0.3 MB | `9735926c15c0d207b5543d6ab3d3780c4bca405d3d60995c4e68d9ade781b940` |
| `mark 1/mark_4b_outputs/probability_cache/volume_114.npz` | 0.2 MB | `2f3949daa3117fbc9a437a7e5666514fab5af43f14a9e29ea717039901daef2b` |
| `mark 1/mark_4b_outputs/probability_cache/volume_115.npz` | 0.2 MB | `5afdba49fd51be1ae2292f053a08f24e2dfbdc6099b7822f2b79956be8986d4c` |
| `mark 1/mark_4b_outputs/probability_cache/volume_116.npz` | 0.2 MB | `b677dd569c795a9d2daad706d6d7e610d1749ef47830bc25639d53df217f5c5c` |
| `mark 1/mark_4d_outputs/probability_cache/control/volume_104.npz` | 0.2 MB | `a241de8afcd6086e7f75cda4b8ec354f6f7894f10e1426e6df062bb6bae5b2dd` |
| `mark 1/mark_4d_outputs/probability_cache/control/volume_105.npz` | 0.2 MB | `260eb563ef9c1f5296e29bb494c9817b923f068268a2d96dd6dd6c2f5b5ee788` |
| `mark 1/mark_4d_outputs/probability_cache/control/volume_106.npz` | 0.2 MB | `6afc5c3f11ff8440a9549eb5fe64d41b70991ec88285d76020a4fb0687230ebd` |
| `mark 1/mark_4d_outputs/probability_cache/control/volume_107.npz` | 0.2 MB | `9d950c866b0048f468d401a441244bc0a9f0a56b166d9bb694af70d08779b6a6` |
| `mark 1/mark_4d_outputs/probability_cache/control/volume_108.npz` | 1.6 MB | `7ca9525b413f65acab7b419e79ebadc9b8cd769ad6a66d9961b6adfaf7dff9ef` |
| `mark 1/mark_4d_outputs/probability_cache/control/volume_109.npz` | 0.3 MB | `100225e61a1bdb7a6948a596596a334e28e470c0c1f167253bfb84f88983518e` |
| `mark 1/mark_4d_outputs/probability_cache/control/volume_110.npz` | 0.3 MB | `cea94a1022fd626720e210ee2c3275f52a602bf45ba6fb988343bee0ef2067d7` |
| `mark 1/mark_4d_outputs/probability_cache/control/volume_111.npz` | 0.2 MB | `61de7873af5957caf8aaece22c60f8334c3a30b585b7651f0eadb116c6cc9a92` |
| `mark 1/mark_4d_outputs/probability_cache/control/volume_112.npz` | 0.2 MB | `f1db962d6efd7d24f8be20f4a7b53bbfc21eae3c1db722335414f6ac28d676ec` |
| `mark 1/mark_4d_outputs/probability_cache/control/volume_113.npz` | 0.3 MB | `678cb84198d46afe299dc5812e4070e8256c716b89ee79226f545d6858979e09` |
| `mark 1/mark_4d_outputs/probability_cache/control/volume_114.npz` | 0.2 MB | `98ab0c3978c07d62bbabd68db412115b59032df3627b5331ed46b82620e0a132` |
| `mark 1/mark_4d_outputs/probability_cache/control/volume_115.npz` | 0.2 MB | `5e3f5f77fe6882d7da7040fe74576dd9213f66a32d8e7ae1726fb8e749c49a84` |
| `mark 1/mark_4d_outputs/probability_cache/control/volume_116.npz` | 0.2 MB | `379a64ef6e644e51d1459b010e456088ec8daea72fffbd170f4dc35190a3c223` |
| `mark 1/mark_4d_outputs/probability_cache/recall_loss/volume_104.npz` | 0.2 MB | `334fc9460ff8b98f2b4e8b685e53babd3752a092b28a82bc8050414e3cb59277` |
| `mark 1/mark_4d_outputs/probability_cache/recall_loss/volume_105.npz` | 0.2 MB | `1273e2b6d517a0e1e5b9ec92929132dc8acb61f8148c553ad396a6642b3e5e9f` |
| `mark 1/mark_4d_outputs/probability_cache/recall_loss/volume_106.npz` | 0.2 MB | `499eb18d3aebb3986187fc7369bb80a3ef20b804cceb417b2fe64b02a215da50` |
| `mark 1/mark_4d_outputs/probability_cache/recall_loss/volume_107.npz` | 0.2 MB | `f4cb0708402192c908c24da53bb0bc2d7461c542c344cc9077dc556d34616980` |
| `mark 1/mark_4d_outputs/probability_cache/recall_loss/volume_108.npz` | 1.2 MB | `cec4eb19eeb64a4c9c7d3fd46f4eb7194837183d950f7918faf6a78e8fc4625c` |
| `mark 1/mark_4d_outputs/probability_cache/recall_loss/volume_109.npz` | 0.3 MB | `e913d65fa756f459a8b41cfffa66c479755a2c4257c04e309fe8c443572d5ed1` |
| `mark 1/mark_4d_outputs/probability_cache/recall_loss/volume_110.npz` | 0.3 MB | `16b1d83b9ca5e6fa715966f05efe9cf14d57dad7e8bd94ea1c64c9076a33e240` |
| `mark 1/mark_4d_outputs/probability_cache/recall_loss/volume_111.npz` | 0.2 MB | `f567d8a9b1c5b274a550b7493c428c3612a38fa64329d219de2c0bf9f3015b5d` |
| `mark 1/mark_4d_outputs/probability_cache/recall_loss/volume_112.npz` | 0.2 MB | `712561eb6e962fe1eddf7822284c72e70ead12a879b8760af95781e44c004409` |
| `mark 1/mark_4d_outputs/probability_cache/recall_loss/volume_113.npz` | 0.3 MB | `8daeeba82c7c6662a96491fca6cf7ca7fbbfbc07de7fb3a3be710d7478a7a709` |
| `mark 1/mark_4d_outputs/probability_cache/recall_loss/volume_114.npz` | 0.2 MB | `5a2c27bc962b04d945109816b244131f584974ef2bbe8cd9ab44e446353cb2f6` |
| `mark 1/mark_4d_outputs/probability_cache/recall_loss/volume_115.npz` | 0.2 MB | `a1e8897f6c1eb5bbd1f7c46ac608bcfb6291e8f0de4d94504b850bcf7a9b2b57` |
| `mark 1/mark_4d_outputs/probability_cache/recall_loss/volume_116.npz` | 0.2 MB | `3fa069087fc62d2c641f51faab33ce842c15a5a689023a2d481f08a88eb086c5` |

## 3. Required checkpoints — still present in working tree (do NOT delete)

| Path | Role |
|---|---|
| `Practice/multitask_liver_tumor_outputs/multitask_best.pth` | kept in working tree |
| `mark 1/mark_4_outputs/mark_4_best.pth` | kept in working tree |
| `mark 1/mark_4c_outputs/recall_loss_best.pth` | kept in working tree |
| `mark 1/mark_4c_outputs/two_channel_best.pth` | kept in working tree |
| `Evaluation/output/03_mark_3/data/broad_1ch_overfit.pth` | kept in working tree |
| `Evaluation/output/03_mark_3/data/broad_liver_2ch_overfit.pth` | kept in working tree |
| `Evaluation/output/03_mark_3/data/broad_liver_narrow_3ch_overfit.pth` | kept in working tree |
| `Evaluation/output/06_mark_4c/data/recall_loss_best.pth` | kept in working tree |
| `Evaluation/output/06_mark_4c/data/two_channel_best.pth` | kept in working tree |
