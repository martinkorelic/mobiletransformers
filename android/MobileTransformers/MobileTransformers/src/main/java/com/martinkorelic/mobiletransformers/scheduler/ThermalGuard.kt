package com.martinkorelic.mobiletransformers.scheduler

import android.content.Context
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager

/**
 * #34: the device-state readings a scheduled chunk records and acts on.
 *
 * The plan's framing is that **the measurement is the contribution**, so this is not only a gate —
 * every chunk emits a [ThermalSample] into the run log, which is what the thermal/energy trace is
 * made of. Keeping the *decision* pure ([shouldPause]) lets it be asserted on a host, while the
 * *reading* stays a thin platform call.
 */
data class ThermalSample(
    /** [PowerManager.getCurrentThermalStatus], or -1 when the platform predates API 29. */
    val thermalStatus: Int,
    /** Battery charge 0..100, or -1 when unavailable. */
    val batteryPercent: Int,
    /** Battery temperature in tenths of a degree C, or -1 when unavailable. */
    val batteryTemperatureDeciC: Int,
    /** Cumulative charge counter in µAh, or -1. Differenced across chunks it gives energy drawn. */
    val chargeCounterMicroAh: Long,
    val timestampMillis: Long,
) {
    /** One CSV row, in the `docs/mobile_evaluation.md` style. */
    fun toCsvRow(chunk: Int, globalStep: Int): String =
        "$timestampMillis,$chunk,$globalStep,$thermalStatus,$batteryPercent," +
            "$batteryTemperatureDeciC,$chargeCounterMicroAh"

    companion object {
        const val CSV_HEADER =
            "timestampMillis,chunk,globalStep,thermalStatus,batteryPercent," +
                "batteryTemperatureDeciC,chargeCounterMicroAh"
    }
}

object ThermalGuard {

    /**
     * Pause the run at [PowerManager.THERMAL_STATUS_SEVERE] or worse.
     *
     * Pure, so the boundary is asserted rather than assumed. `-1` (pre-API-29, no reading) does NOT
     * pause: refusing to train because the platform is too old to tell us the temperature would make
     * the feature unavailable on exactly the devices it is meant for.
     */
    fun shouldPause(thermalStatus: Int): Boolean =
        thermalStatus >= PowerManager.THERMAL_STATUS_SEVERE

    fun sample(context: Context, nowMillis: Long = System.currentTimeMillis()): ThermalSample {
        val power = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
        val battery = context.getSystemService(Context.BATTERY_SERVICE) as? BatteryManager

        val thermal =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && power != null) {
                power.currentThermalStatus
            } else {
                -1
            }

        return ThermalSample(
            thermalStatus = thermal,
            batteryPercent = battery?.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY) ?: -1,
            batteryTemperatureDeciC = readBatteryTemperature(context),
            chargeCounterMicroAh =
                battery?.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER) ?: -1L,
            timestampMillis = nowMillis,
        )
    }

    private fun readBatteryTemperature(context: Context): Int {
        val intent =
            context.registerReceiver(null, android.content.IntentFilter(android.content.Intent.ACTION_BATTERY_CHANGED))
        return intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1
    }
}
