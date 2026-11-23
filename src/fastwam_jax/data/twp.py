"""TWP3 episode packing for the RoboTwin training loader.

The little-endian header and section table describe frame-major JPEGs,
frame-aligned actions and measured states, and episode instructions. Text
embeddings may be empty when a separate table supplies them. The training
loader reads through the JPEG payload and leaves held-out instructions unread.
"""
from __future__ import annotations

import struct
from typing import Sequence

import numpy as np

MAGIC = b"TWP3"
VERSION = 3
PREAMBLE = 40
ENTRY = 20                      # 段表一条 = tag u32 + offset u64 + length u64

LENGTHS, ACTIONS, SEEN, PAYLOAD, UNSEEN, TEXT_EMB, STATES = 1, 2, 3, 4, 5, 6, 7
SECTION_NAMES = {LENGTHS: "lengths", ACTIONS: "actions", SEEN: "instr_seen",
                 TEXT_EMB: "text_emb", PAYLOAD: "payload", UNSEEN: "instr_unseen",
                 STATES: "states"}


def validate_alignment(spec: dict) -> None:
    """Require RGB pixels and frame-aligned action and state rows."""
    if spec.get("color_order", "rgb") != "rgb":
        raise ValueError(f"TWP3 requires RGB images, got color_order={spec['color_order']!r}")
    if spec.get("action_shift", 0) != 0:
        raise ValueError(
            f"TWP3 requires aligned action and state rows, got action_shift={spec['action_shift']!r}"
        )


# ---------------------------------------------------------------- 写

def pack_strings(items: Sequence[str]) -> bytes:
    """count u32 | length u32[count] | UTF-8 拼接。定长表 + 裸字节,不是序列化容器。"""
    enc = [s.encode("utf-8") for s in items]
    return (struct.pack("<I", len(enc))
            + b"".join(struct.pack("<I", len(e)) for e in enc)
            + b"".join(enc))


def build(episode_id: int, jpeg_lengths, jpeg_payload: bytes,
          actions: np.ndarray, seen: Sequence[str], unseen: Sequence[str],
          *, n_cams: int, states: np.ndarray, text_emb: np.ndarray | None = None) -> bytes:
    """Pack a single episode from frame-major JPEGs and aligned numeric streams."""
    lengths = np.asarray(jpeg_lengths, dtype=np.int32)
    acts = np.ascontiguousarray(actions, dtype=np.float32)
    if acts.ndim != 2:
        raise ValueError(f"actions must be [T, D], got {acts.shape}")
    state = np.ascontiguousarray(states, dtype=np.float32)
    if state.shape != acts.shape:
        raise ValueError(f"states must match actions {acts.shape}, got {state.shape}")
    T, dim = acts.shape
    if len(lengths) != T * n_cams:
        raise ValueError(f"lengths {len(lengths)} != T*C = {T}*{n_cams}")
    if int(lengths.sum()) != len(jpeg_payload):
        raise ValueError(f"lengths 和 {lengths.sum()} != payload {len(jpeg_payload)}")
    if text_emb is None:
        t_rows = t_len = t_dim = 0
        t_blob = b""
    else:
        te = np.ascontiguousarray(text_emb, dtype=np.float16)
        if te.ndim != 3:
            raise ValueError(f"text_emb 应为 (rows, len, dim),实为 {te.shape}")
        t_rows, t_len, t_dim = te.shape
        t_blob = te.tobytes()

    # 段序:TEXT_EMB 必须在 PAYLOAD **之前** —— 训练侧只读 [0, payload_end),
    # 放后面就读不到,放到 UNSEEN 之后更是会把评测指令一起拖进训练进程。
    secs = [
        (LENGTHS, lengths.tobytes()), (ACTIONS, acts.tobytes()), (STATES, state.tobytes()),
        (SEEN, pack_strings(seen)), (TEXT_EMB, t_blob), (PAYLOAD, jpeg_payload),
        (UNSEEN, pack_strings(unseen)),
    ]
    pre = MAGIC + struct.pack("<IIIIIIIII", VERSION, len(secs), episode_id, T, n_cams, dim,
                              t_rows, t_len, t_dim)
    off = PREAMBLE + len(secs) * ENTRY
    table, body = b"", b""
    for tag, blob in secs:
        table += struct.pack("<IQQ", tag, off, len(blob))
        body += blob
        off += len(blob)
    return pre + table + body


# ---------------------------------------------------------------- 读

class Episode:
    """对一段 .twp 字节的零拷贝视图。切片全是指针运算,不复制不解码。"""

    __slots__ = ("buf", "episode_id", "version", "n_frames", "n_cams", "action_dim",
                 "text_rows", "text_len", "text_dim", "_sec", "_cum", "_seen")

    def __init__(self, buf, *, expect_id: int | None = None):
        if bytes(buf[:4]) != MAGIC:
            raise ValueError(f"magic 不符: {bytes(buf[:4])!r}(需要 {MAGIC!r})")
        (self.version, n_sec, self.episode_id, self.n_frames, self.n_cams,
         self.action_dim, self.text_rows, self.text_len,
         self.text_dim) = struct.unpack("<IIIIIIIII", buf[4:PREAMBLE])
        if self.version != VERSION:
            raise ValueError(f"版本 {self.version} != {VERSION}")
        # 抓"集错位":文件名对但内容是别的集。判定门之外的运行时防线
        if expect_id is not None and self.episode_id != expect_id:
            raise ValueError(f"集号错位:文件内 {self.episode_id},期望 {expect_id}")
        self.buf = buf
        self._sec = {}
        for k in range(n_sec):
            tag, off, ln = struct.unpack("<IQQ", buf[PREAMBLE + k*ENTRY: PREAMBLE + (k+1)*ENTRY])
            if tag in self._sec:
                raise ValueError(f"ep{self.episode_id}: 重复段 tag={tag}")
            self._sec[tag] = (off, ln)
        missing = set(SECTION_NAMES) - self._sec.keys()
        if missing:
            names = ", ".join(SECTION_NAMES[tag] for tag in sorted(missing))
            raise ValueError(f"episode {self.episode_id}: missing TWP3 sections: {names}")
        self._cum = None
        self._seen = None
        # 形状自洽:preamble 说的和段长必须对得上,否则文件是坏的
        if self._sec[ACTIONS][1] != self.n_frames * self.action_dim * 4:
            raise ValueError(f"ep{self.episode_id}: actions 段 {self._sec[ACTIONS][1]}B "
                             f"!= T*dim*4 = {self.n_frames}*{self.action_dim}*4")
        if self._sec[STATES][1] != self.n_frames * self.action_dim * 4:
            raise ValueError(f"ep{self.episode_id}: states 段 {self._sec[STATES][1]}B "
                             f"!= T*dim*4 = {self.n_frames}*{self.action_dim}*4")
        if self._sec[LENGTHS][1] != self.n_frames * self.n_cams * 4:
            raise ValueError(f"ep{self.episode_id}: lengths 段 {self._sec[LENGTHS][1]}B "
                             f"!= T*C*4 = {self.n_frames}*{self.n_cams}*4")
        want_t = self.text_rows * self.text_len * self.text_dim * 2
        if self._sec[TEXT_EMB][1] != want_t:
            raise ValueError(f"ep{self.episode_id}: text_emb 段 {self._sec[TEXT_EMB][1]}B "
                             f"!= rows*len*dim*2 = {self.text_rows}*{self.text_len}"
                             f"*{self.text_dim}*2")
        # TEXT_EMB 必须在 PAYLOAD 之前,否则训练侧的一次读取拿不到它
        if self._sec[TEXT_EMB][0] > self._sec[PAYLOAD][0]:
            raise ValueError(f"ep{self.episode_id}: text_emb 段在 payload 之后 —— "
                             f"训练侧只读 [0,payload_end),会读不到")
        if self._sec[STATES][0] > self._sec[PAYLOAD][0]:
            raise ValueError(f"ep{self.episode_id}: states 段在 payload 之后 —— "
                             f"训练侧只读 [0,payload_end),会读不到")

    # -------- 段访问

    def _raw(self, tag: int) -> memoryview:
        off, ln = self._sec[tag]
        return memoryview(self.buf)[off:off + ln]

    @property
    def lengths(self) -> np.ndarray:
        off, ln = self._sec[LENGTHS]
        return np.frombuffer(self.buf, np.int32, ln // 4, off)

    @property
    def actions(self) -> np.ndarray:
        """控制器下发的**原值**。delta 与归一化不在这里做。"""
        off, _ = self._sec[ACTIONS]
        return np.frombuffer(self.buf, np.float32,
                             self.n_frames * self.action_dim, off).reshape(self.n_frames, self.action_dim)

    @property
    def states(self) -> np.ndarray:
        """Measured proprioception aligned to observation frames."""
        off, _ = self._sec[STATES]
        return np.frombuffer(self.buf, np.float32,
                             self.n_frames * self.action_dim, off).reshape(self.n_frames, self.action_dim)

    @property
    def seen(self) -> list[str]:
        # 缓存:一集在池里活很久,每步每被抽到都重解 100 条串是白烧 CPU
        if self._seen is None:
            self._seen = unpack_strings(self._raw(SEEN))
        return self._seen

    @property
    def unseen(self) -> list[str]:
        """**训练路径禁止调用。** 官方规定训练用 seen、评测用 unseen;
        混用会毁掉语言泛化基准,而且毁了之后无法自证清白。"""
        return unpack_strings(self._raw(UNSEEN))

    @property
    def text_emb(self) -> np.ndarray:
        """[text_rows, text_len, text_dim] fp16。零拷贝视图,不复制。

        没有文本段(或长度 0)时返回形状 (0,0,0) 的空数组,让调用方能明确判断,
        而不是拿到一个看起来正常的零向量。
        """
        if self.text_rows == 0:
            return np.empty((0, 0, 0), np.float16)
        off, ln = self._sec[TEXT_EMB]
        return np.frombuffer(self.buf, np.float16, ln // 2, off).reshape(
            self.text_rows, self.text_len, self.text_dim)

    def payload_end(self) -> int:
        """训练侧应当只 range-read [0, payload_end)。unseen 段在此之后。"""
        off, ln = self._sec[PAYLOAD]
        return off + ln

    # -------- 帧访问

    def _offsets(self) -> np.ndarray:
        if self._cum is None:                       # 每集一次,微秒级
            self._cum = np.concatenate([[0], np.cumsum(self.lengths, dtype=np.int64)])
        return self._cum

    def jpeg(self, frame: int, cam: int) -> memoryview:
        """第 frame 帧第 cam 路的 JPEG 原始字节(零拷贝)。"""
        base = self._sec[PAYLOAD][0]
        cum = self._offsets()
        k = frame * self.n_cams + cam
        return memoryview(self.buf)[base + cum[k]: base + cum[k + 1]]


def unpack_strings(b) -> list[str]:
    b = bytes(b)
    n = struct.unpack("<I", b[:4])[0]
    lens = struct.unpack(f"<{n}I", b[4:4 + 4 * n])
    out, p = [], 4 + 4 * n
    for L in lens:
        out.append(b[p:p + L].decode("utf-8"))
        p += L
    return out
