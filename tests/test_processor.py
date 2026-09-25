"""The processor applies exactly the labels released before each transaction, whatever
order it reads the two topics in."""

from fraud.stream.messages import decode, encode
from fraud.stream.processor import run

TOPICS = {"transactions": "tx", "labels": "lb", "decisions": "dec"}


class Msg:
    def __init__(self, topic, value):
        self._topic, self._value = topic, encode(value)

    def topic(self):
        return self._topic

    def key(self):
        return b"k"

    def value(self):
        return self._value

    def error(self):
        return None


class Consumer:
    def __init__(self, messages):
        self.messages = list(messages)

    def subscribe(self, topics):
        assert set(topics) == {"tx", "lb"}

    def poll(self, timeout):
        return self.messages.pop(0) if self.messages else None

    def close(self):
        pass


class Producer:
    def __init__(self):
        self.sent = []

    def produce(self, topic, key, value):
        self.sent.append(decode(value))

    def poll(self, timeout):
        pass

    def flush(self):
        pass


class Scorer:
    """Records how many labels had been applied when each transaction was scored."""

    class _Features:
        def state_size(self):
            return {}

    def __init__(self):
        self.applied = []
        self.features = self._Features()

    def observe_label(self, label):
        self.applied.append(label["TransactionID"])

    def decide(self, event):
        assert "_labels_before" not in event and "_sent_at" not in event
        return {
            "TransactionID": event["TransactionID"],
            "labels_seen": list(self.applied),
            "timing_ms": {"total": 1.0},
        }


def _stream(order: str):
    # three transactions; two labels go out before the second and one before the third
    txs = [
        {"TransactionID": 10, "_labels_before": 0},
        {"TransactionID": 11, "_labels_before": 2},
        {"TransactionID": 12, "_labels_before": 3},
    ]
    labels = [{"TransactionID": i} for i in (1, 2, 3, 4)]
    tx_msgs = [Msg("tx", t) for t in txs]
    lb_msgs = [Msg("lb", x) for x in labels]
    if order == "labels_first":
        return lb_msgs + tx_msgs
    if order == "transactions_first":
        return tx_msgs + lb_msgs
    return [lb_msgs[0], tx_msgs[0], tx_msgs[1], lb_msgs[1], lb_msgs[2], tx_msgs[2], lb_msgs[3]]


def test_each_transaction_sees_exactly_its_labels_in_any_read_order():
    for order in ("labels_first", "transactions_first", "interleaved"):
        producer = Producer()
        stats = run(Consumer(_stream(order)), producer, Scorer(), TOPICS, idle_seconds=0.0)
        seen = {d["TransactionID"]: d["labels_seen"] for d in producer.sent}
        assert seen == {10: [], 11: [1, 2], 12: [1, 2, 3]}, order
        assert stats["events"] == 3
