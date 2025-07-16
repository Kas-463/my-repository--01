package com.example.myapplication; // 替换为你的包名

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Environment;
import android.os.Handler;
import android.util.Log;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Calendar;
import java.util.Locale;

public class SensorDataLogger implements SensorEventListener {
    private static final String TAG = "SensorDataLogger";
    private static final int SAMPLE_RATE_HZ = 50; // 50Hz采样频率
    private static final int SAMPLE_INTERVAL_MS = 1000 / SAMPLE_RATE_HZ; // 20ms

    // 单下划线分隔时间单位
    private static final String TIME_FORMAT = "%04d_%04d_%04d_%04d_%04d_%04d_%04d";

    private final SensorManager sensorManager;
    private final Sensor sensor;
    private BufferedWriter bufferedWriter;
    private boolean isLogging = false;
    private final Handler handler = new Handler();
    private float[] sensorValues = {0, 0, 0};
    private long lastSensorTimestamp;
    private int writeCount = 0;

    public SensorDataLogger(Context context) {
        sensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        sensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
    }

    public void startLogging() {
        if (isLogging) return;

        try {
            File file = createOutputFile();
            FileWriter fileWriter = new FileWriter(file, true);
            bufferedWriter = new BufferedWriter(fileWriter);

            // 写入CSV表头
            bufferedWriter.write("Timestamp,X_Value,Y_Value,Z_Value\n");
            bufferedWriter.flush();

            isLogging = true;
            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_FASTEST);
            handler.post(loggingRunnable);

        } catch (IOException e) {
            Log.e(TAG, "Error creating file", e);
        }
    }

    public void stopLogging() {
        if (!isLogging) return;

        isLogging = false;
        sensorManager.unregisterListener(this);
        handler.removeCallbacks(loggingRunnable);

        try {
            if (bufferedWriter != null) {
                bufferedWriter.flush();
                bufferedWriter.close();
            }
        } catch (IOException e) {
            Log.e(TAG, "Error closing file", e);
        }
    }

    private File createOutputFile() {
        File storageDir = new File(
                Environment.getExternalStorageDirectory(),
                "Android/data/" + "com.example.myapplication" + "/SensorData"
        );

        if (!storageDir.exists() && !storageDir.mkdirs()) {
            Log.e(TAG, "Failed to create directory: " + storageDir.getAbsolutePath());
        }

        Calendar cal = Calendar.getInstance();
        return new File(storageDir, String.format(Locale.US,
                "sensor_%04d%02d%02d_%02d%02d.csv",
                cal.get(Calendar.YEAR),
                cal.get(Calendar.MONTH) + 1,
                cal.get(Calendar.DAY_OF_MONTH),
                cal.get(Calendar.HOUR_OF_DAY),
                cal.get(Calendar.MINUTE)));
    }

    private final Runnable loggingRunnable = new Runnable() {
        @Override
        public void run() {
            if (isLogging) {
                writeSensorData();
                handler.postDelayed(this, SAMPLE_INTERVAL_MS);
            }
        }
    };

    private void writeSensorData() {
        try {
            // 使用传感器事件的实际时间戳
            String timestamp = formatTimestamp(lastSensorTimestamp);

            // 格式化传感器数据（6位小数）
            String dataLine = String.format(Locale.US, "%s,%.6f,%.6f,%.6f%n",
                    timestamp, sensorValues[0], sensorValues[1], sensorValues[2]);

            bufferedWriter.write(dataLine);

            // 每100条数据刷一次盘，提高效率
            if (++writeCount % 100 == 0) {
                bufferedWriter.flush();
            }

        } catch (IOException e) {
            Log.e(TAG, "Error writing data", e);
        }
    }

    private String formatTimestamp(long nanosTimestamp) {
        // 将纳秒时间戳转换为毫秒
        long millis = nanosTimestamp / 1_000_000;

        Calendar cal = Calendar.getInstance();
        cal.setTimeInMillis(millis);

        return String.format(Locale.US, TIME_FORMAT,
                cal.get(Calendar.YEAR),
                cal.get(Calendar.MONTH) + 1,
                cal.get(Calendar.DAY_OF_MONTH),
                cal.get(Calendar.HOUR_OF_DAY),
                cal.get(Calendar.MINUTE),
                cal.get(Calendar.SECOND),
                cal.get(Calendar.MILLISECOND));
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        System.arraycopy(event.values, 0, sensorValues, 0, event.values.length);
        lastSensorTimestamp = event.timestamp;
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // 不需要实现
    }
}