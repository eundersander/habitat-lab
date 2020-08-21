#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""todo
"""
import argparse
import glob
import os
import sqlite3

from habitat.utils import compare_profiles


def merge_events(event_lists):

    # When merging profiles from different machines, there are two problems related to thread ids.
    # 1, there could be duplicates
    # 2, when viewing profiles, it would be nice to understand which threads are from the same machine.

    # We remap the original thread ids as follows:
    #   first process:  1000, 1001, 1002, ...
    #   second process: 2000, 2001, 2002, ...
    merged_events = []
    for list_index, list in enumerate(event_lists):
        thread_id_remap = {}
        next_thread_id = (list_index + 1) * 1000
        for event in list:
            if event._thread_id not in thread_id_remap:
                thread_id_remap[event._thread_id] = next_thread_id
                next_thread_id += 1
        for event in list:
            remapped_event = compare_profiles.Event(
                event._name,
                thread_id_remap[event._thread_id],
                event._start,
                event._end,
            )
            merged_events.append(remapped_event)

    return merged_events


def write_sqlite_database(events, output_conn):
    # This create statement corresponds to the sqlite database that Nsight Nsys
    # creates.
    output_conn.execute(
        """CREATE TABLE NVTX_EVENTS (start INTEGER NOT NULL, end INTEGER, text TEXT, globalTid INTEGER)"""
    )

    for event in events:
        output_conn.execute(
            "INSERT INTO NVTX_EVENTS VALUES ({}, {}, '{}', {})".format(
                event._start, event._end, event._name, event._thread_id
            )
        )

    output_conn.commit()


def create_arg_parser():
    """For compare_profiles.py script. Includes print formatting options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("filepaths", nargs="+")
    parser.add_argument("--output")
    return parser


def main():
    args = create_arg_parser().parse_args()

    if os.path.exists(args.output):
        print("error: output {} already exists".format(args.output))
        return

    event_lists = []
    for filepath in args.filepaths:
        event_lists.append(
            compare_profiles.get_sqlite_events(sqlite3.connect(filepath))
        )

    merged_events = merge_events(event_lists)

    output_conn = sqlite3.connect(args.output)
    write_sqlite_database(merged_events, output_conn)
    output_conn.close()

    print("merged {} profiles to {}".format(len(args.filepaths), args.output))


if __name__ == "__main__":
    main()
