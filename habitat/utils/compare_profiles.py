
import sqlite3
import glob, os
import argparse

class Event:
    def __init__(self, name, thread_id, start, end):
        self._name = name
        self._thread_id = thread_id
        self._start = start
        self._end = end


class SummaryItem:
    def __init__(self):
        self._time_exclusive = 0
        self._time_inclusive = 0
    

def get_sqlite_events(conn):

    cursor = conn.execute("SELECT text, globalTid, start, end from NVTX_EVENTS")

    events = []

    for row in cursor:
        events.append(Event(row[0], row[1], row[2], row[3]))

    assert len(events) > 0, "We failed to parse any events from {}".format(filepath)

    return events


def create_summary_from_events(events):

    # sort by start time (ascending). For ties, sort by end time (descending). In this way,
    # a (shorter) child event that starts at the same time as a (longer) parent event
    # will occur later in the sort.
    events.sort(reverse=True, key=lambda event: event._end)
    events.sort(reverse=False, key=lambda event: event._start)

    items = {}

    for i in range(len(events)):

        event = events[i]

        item = items.setdefault(event._name, SummaryItem())

        event_duration = event._end - event._start
        item._time_inclusive += event_duration

        exclusive_duration = 0

        # iterate chronologically through later events. Our accumulated "exclusive
        #  duration" is time during which we aren't inside any overlapping, same-thread 
        # event ("child event").
        recent_exclusive_start_time = event._start
        child_end_times = set()
        for j in range(i + 1, len(events) + 1):  # sloppy: one extra iteration

            other_event = None if j == len(events) else events[j]
            if other_event:
                if other_event._thread_id != event._thread_id:
                    continue
                if other_event._start > event._end:
                    other_event = None

            current_time = other_event._start if other_event else event._end

            if len(child_end_times):
                latest_child_end_time = max(child_end_times)

                # remove any child which ends prior to current_time
                child_end_times = set(filter(lambda t : t > current_time, child_end_times))

                if len(child_end_times) == 0:
                    recent_exclusive_start_time = latest_child_end_time
                else:
                    recent_exclusive_start_time = None

            # handle exclusive time leading up to current_time
            if recent_exclusive_start_time:
                assert recent_exclusive_start_time <= current_time
                exclusive_duration += current_time - recent_exclusive_start_time

            if other_event:
                child_end_times.add(other_event._end)
            else:
                break

        assert event_duration >= exclusive_duration
        item._time_exclusive += exclusive_duration

        items[event._name] = item

    return items


def display_time_ms(time, show_sign=False):

    # assume times from profiles are nanoseconds
    return "{}{:,.0f}".format(
        "+" if time > 0 and show_sign else "",
        time / (1000 * 1000))
    # return time

def print_summaries(summaries, args, labels=None):

    sort_by_exclusive = args.sort_by == "exclusive"
    print_relative_timings = args.relative

    if len(summaries) == 0:
        print("no summaries to print")

    all_names_with_times = {}
    max_name_len = 0
    for summary in summaries:
        for name in summary:
            all_names_with_times.setdefault(name,
                summary[name]._time_exclusive if sort_by_exclusive else
                summary[name]._time_inclusive)
            max_name_len = max(max_name_len, len(name))

    all_names_with_times_list = list(all_names_with_times.items())
    # sort by time, decreasing
    all_names_with_times_list.sort(reverse=True, key=lambda x : x[1])

    column_pad = 2
    time_width = 12

    if labels:
        assert(len(labels) == len(summaries))
        max_label_len = time_width * 2 + column_pad
        print("".ljust(max_name_len + column_pad), end='')
        for label in labels:
            short_label = label[-max_label_len:]
            print(short_label.ljust(max_label_len + column_pad), end='')
        print("")

    print("event name".ljust(max_name_len + column_pad) 
        + "incl (ms)".rjust(time_width).ljust(time_width + column_pad)
        + "excl (ms)".rjust(time_width).ljust(time_width + column_pad))

    for tup in all_names_with_times_list:

        name = tup[0]
        print(name.ljust(max_name_len + column_pad), end='')

        for (index, summary) in enumerate(summaries):

            base_summary = summaries[0] if index > 0 else None

            if name in summary:

                item = summary[name]

                if base_summary and print_relative_timings and name in base_summary:

                    base_item = base_summary[name]
                    time_inclusive = item._time_inclusive - base_item._time_inclusive
                    time_exclusive = item._time_exclusive - base_item._time_exclusive
                    show_sign = True
                else:
                    time_inclusive = item._time_inclusive
                    time_exclusive = item._time_exclusive
                    show_sign = False

                print(display_time_ms(time_inclusive, show_sign=show_sign).rjust(time_width).ljust(time_width + column_pad)
                    + display_time_ms(time_exclusive, show_sign=show_sign).rjust(time_width).ljust(time_width + column_pad),
                    end='')
                


            else:
                print("-".ljust(time_width + column_pad)
                    + "-".ljust(time_width + column_pad),
                    end='')

        print("")
    

def get_sqlite_filepaths_from_directory(directory):

    filepaths = []

    os.chdir(directory)
    for filepath in glob.glob("*.sqlite"):
        filepaths.append(filepath)

    return filepaths


def test():

    c = sqlite3.connect(':memory:')

    # this create statement corresponds to the sqlite database that Nsight Nsys creates.
    c.execute('''CREATE TABLE NVTX_EVENTS (start INTEGER NOT NULL, end INTEGER, text TEXT, globalTid INTEGER)''')    

    # Insert some events
    # Thread 1
    # 01234567890
    # .[        )   incl 90   excl 20  root A
    # .[ ).......   incl 20   excl 10  child 0
    # ..[).......   incl 10   excl 10  child 1
    # ....[...)..   incl 40   excl 10  child 2
    # .....[...).   incl 40   excl 30  child 3
    # ......[)...   incl 10   excl 10  child 4

    # Thread 2
    # 01234567890
    # ..[......).   incl 70   excl 10  root B
    # ...[.....).   incl 60   excl 60  child 6

    # note on child 3: it's unrealistic for a child event to extend beyond the end time
    # of its parent (child 2 in this case), but we expect to handle it anyway.

    # events are numbered here based on chronological ordering, but we insert them
    # in random order.
    c.execute("INSERT INTO NVTX_EVENTS VALUES (60, 70, 'child 4', 1)")
    c.execute("INSERT INTO NVTX_EVENTS VALUES (20, 30, 'child 1', 1)")
    c.execute("INSERT INTO NVTX_EVENTS VALUES (10, 30, 'child 0', 1)")
    c.execute("INSERT INTO NVTX_EVENTS VALUES (10, 100, 'root A', 1)")
    c.execute("INSERT INTO NVTX_EVENTS VALUES (50, 90, 'child 3', 1)")
    c.execute("INSERT INTO NVTX_EVENTS VALUES (40, 80, 'child 2', 1)")

    c.execute("INSERT INTO NVTX_EVENTS VALUES (20, 90, 'root B', 2)")
    c.execute("INSERT INTO NVTX_EVENTS VALUES (30, 90, 'child 6', 2)")

    # Save (commit) the changes
    c.commit()    

    events = get_sqlite_events(c)
    # print("events: {}".format(events))
    summary = create_summary_from_events(events)

    assert summary["root A"]._time_inclusive == 90
    assert summary["root A"]._time_exclusive == 20
    assert summary["child 0"]._time_inclusive == 20
    assert summary["child 0"]._time_exclusive == 10
    assert summary["child 1"]._time_inclusive == 10
    assert summary["child 1"]._time_exclusive == 10
    assert summary["child 2"]._time_inclusive == 40
    assert summary["child 2"]._time_exclusive == 10
    assert summary["child 3"]._time_inclusive == 40
    assert summary["child 3"]._time_exclusive == 30
    assert summary["child 4"]._time_inclusive == 10
    assert summary["child 4"]._time_exclusive == 10

    assert summary["root B"]._time_inclusive == 70
    assert summary["root B"]._time_exclusive == 10
    assert summary["child 6"]._time_inclusive == 60
    assert summary["child 6"]._time_exclusive == 60

    print_summaries([summary])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sort-by",
        default="inclusive",
        choices=["inclusive", "exclusive"],
    )
    parser.add_argument('--relative', action='store_true', default=False)
    args = parser.parse_args()

    # test()

    # temp hard-code
    filepaths = get_sqlite_filepaths_from_directory("./")

    filepaths.sort()  # sort alphabetically; todo: sort by creation date

    summaries = []

    # todo: order by creation date

    for filepath in filepaths:
        # print("filepath: {}".format(filepath))

        events = get_sqlite_events(sqlite3.connect(filepath))
        # print("len(events): {}".format(len(events)))
        # print("events[0]: {} {} {}".format(events[0]._name, events[0]._start, events[0]._end))

        summaries.append(create_summary_from_events(events))

    print_summaries(summaries, args, labels=filepaths)

if __name__ == "__main__":
    main()
