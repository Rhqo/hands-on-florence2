# Hands-on Florence-2

**Captioning**

```bash
uv run main.py --task cap
```

**Obejct Detection**

```bash
uv run main.py --task ob
```

**Region Proposal**

```bash
uv run main.py --task rp
```

**Open-Vocabulary Object Detection**

`--text_input` 뒤에 원하는 text 입력 가능
```bash
uv run main.py --task ov-od --text_input kitty
```

**Cascaded (Captioning + Visual Grounding)**

`--level` 뒤에 0,1,2 입력, 객체의 세밀함 정도 결정
```bash
uv run main.py --task cascaded --level 0
```