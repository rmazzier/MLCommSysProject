import datetime
import enum


class SPLIT(enum.Enum):
    TRAIN = "train"
    VALIDATION = "valid"
    TEST = "test"


def get_date_from_overalltime(overalltime):
    # Set the starting date as 29 june 2016 using the datetime module
    start_date = datetime.datetime(2016, 6, 29)

    # Get the first date which is the starting date plus the number of seconds
    date = start_date + datetime.timedelta(seconds=overalltime)
    return date
