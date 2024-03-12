import datetime


def get_timestamp():
    t = datetime.datetime.now()
    return '{}-{:02d}-{:02d} {:02d}:{:02d}:{:02d},{}'.format(
        t.year,
        t.month,
        t.day,
        t.hour,
        t.minute,
        t.second,
        t.microsecond // 1000
    )
