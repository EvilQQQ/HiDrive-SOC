"""Entry point for the HiDrive single-day SOC calculation."""

from RunOnOrOff import BREVO_RunTrips


BATTERY_CAPACITY_AH = 198.5
INITIAL_SOC = 1.0
TRIP_FILE = "DayTrips/Trip_File_2023-06-17.csv"


def HiDrive_SOC():
    """Run one day of SOC simulation with regenerative braking enabled."""
    return BREVO_RunTrips(
        BTMSOnOrOff="Off",
        HVACOnOrOff="Off",
        RBSOnOrOff="On",
        initial_soc=INITIAL_SOC,
        capacity_ah=BATTERY_CAPACITY_AH,
        trip_file=TRIP_FILE,
    )


if __name__ == "__main__":
    HiDrive_SOC()
