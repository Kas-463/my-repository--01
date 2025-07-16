package com.example.myapplication

import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Color
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterApi.Options
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.locks.ReentrantLock
import kotlin.math.sqrt

class MainActivity : AppCompatActivity(), SensorEventListener {

    // 传感器管理器
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null

    // UI元素
    private lateinit var accelerometerTextView: TextView
    private lateinit var gyroscopeTextView: TextView
    private lateinit var statusTextView: TextView
    private lateinit var predictionTextView: TextView
    private lateinit var resultTextView: TextView

    // 运动状态按钮
    private lateinit var btnStatic: Button
    private lateinit var btnWalking: Button
    private lateinit var btnUpstairs: Button
    private lateinit var btnDownstairs: Button
    private lateinit var btnUphill: Button
    private lateinit var btnRunning: Button

    // 运动状态常量
    private companion object {
        const val STATE_IDLE = "IDLE"
        const val STATE_STATIC = "Static"
        const val STATE_WALKING = "Walking"
        const val STATE_UPSTAIRS = "Upstairs"
        const val STATE_DOWNSTAIRS = "Downstairs"
        const val STATE_UPHILL = "Uphill"
        const val STATE_RUNNING = "Running"
        const val SENSOR_PERMISSION_REQUEST_CODE = 100

        // 新增：采集时间常量
        const val MIN_COLLECTION_DURATION = 3000L // 最少采集3秒

        // 采样率控制（50Hz）
        const val SENSOR_INTERVAL_MS = 20L // 20毫秒 = 50Hz
    }

    // 当前记录状态
    private var currentState = STATE_IDLE
    private var isPredicting = false
    private var predictionStartTime: Long = 0
    private var lastSensorUpdateTime: Long = 0

    // 传感器数据缓存
    private var accelerometerData = FloatArray(3)
    private var gyroscopeData = FloatArray(3)

    // 运动分类器
    private lateinit var motionClassifier: MotionClassifier

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        try {
            // 初始化UI元素
            accelerometerTextView = findViewById(R.id.accelerometer_textview)
            gyroscopeTextView = findViewById(R.id.gyroscope_textview)
            statusTextView = findViewById(R.id.status_textview)
            predictionTextView = findViewById(R.id.prediction_text)
            resultTextView = findViewById(R.id.result_textview)

            // 初始化按钮
            btnStatic = findViewById(R.id.btn_static)
            btnWalking = findViewById(R.id.btn_walking)
            btnUpstairs = findViewById(R.id.btn_upstairs)
            btnDownstairs = findViewById(R.id.btn_downstairs)
            btnUphill = findViewById(R.id.btn_uphill)
            btnRunning = findViewById(R.id.btn_running)

            // 设置按钮点击事件
            btnStatic.setOnClickListener { toggleRecordingState(STATE_STATIC) }
            btnWalking.setOnClickListener { toggleRecordingState(STATE_WALKING) }
            btnUpstairs.setOnClickListener { toggleRecordingState(STATE_UPSTAIRS) }
            btnDownstairs.setOnClickListener { toggleRecordingState(STATE_DOWNSTAIRS) }
            btnUphill.setOnClickListener { toggleRecordingState(STATE_UPHILL) }
            btnRunning.setOnClickListener { toggleRecordingState(STATE_RUNNING) }

            // 初始化传感器管理器
            sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager

            // 检查传感器权限
            if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.BODY_SENSORS)
                != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(android.Manifest.permission.BODY_SENSORS),
                    SENSOR_PERMISSION_REQUEST_CODE
                )
            } else {
                initSensors()
            }

            // 初始UI状态
            updateButtonStates()

            // 初始化运动分类器（在后台线程）
            lifecycleScope.launch(Dispatchers.IO) {
                try {
                    motionClassifier = MotionClassifier(applicationContext)
                    withContext(Dispatchers.Main) {
                        predictionTextView.text = "模型加载成功！开始检测运动状态"
                        statusTextView.text = "传感器已就绪"
                        isPredicting = true
                    }
                } catch (e: Exception) {
                    Log.e("MainActivity", "分类器初始化失败", e)
                    withContext(Dispatchers.Main) {
                        predictionTextView.text = "模型加载失败: ${e.message?.substring(0, 20) ?: "未知错误"}"
                        statusTextView.text = "模型初始化错误"
                        Toast.makeText(
                            this@MainActivity,
                            "模型加载失败: ${e.message?.substring(0, 20) ?: "未知错误"}",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "初始化失败", e)
            Toast.makeText(this, "应用初始化失败: ${e.message?.substring(0, 20) ?: "未知错误"}", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == SENSOR_PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                initSensors()
            } else {
                Toast.makeText(
                    this,
                    "需要传感器权限才能使用此应用",
                    Toast.LENGTH_LONG
                ).show()
                finish()
            }
        }
    }

    private fun initSensors() {
        try {
            // 获取传感器
            accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
            gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

            if (accelerometer == null || gyroscope == null) {
                Toast.makeText(
                    this,
                    "设备缺少必要的传感器",
                    Toast.LENGTH_LONG
                ).show()
                finish()
            } else {
                // 设置精确的50Hz采样率
                val samplingPeriodUs = 20000 // 20 ms = 50 Hz
                sensorManager.registerListener(this, accelerometer, samplingPeriodUs)
                sensorManager.registerListener(this, gyroscope, samplingPeriodUs)
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "传感器初始化失败", e)
            Toast.makeText(this, "传感器初始化失败: ${e.message?.substring(0, 20)}", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onResume() {
        super.onResume()
        // 重新注册传感器监听
        try {
            accelerometer?.let {
                sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST)
            }
            gyroscope?.let {
                sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST)
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "传感器注册失败", e)
        }
    }

    override fun onPause() {
        super.onPause()
        // 确保注销监听
        try {
            sensorManager.unregisterListener(this)
        } catch (e: Exception) {
            Log.e("MainActivity", "传感器注销失败", e)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            sensorManager.unregisterListener(this)
            if (::motionClassifier.isInitialized) {
                motionClassifier.close()
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "模型关闭失败", e)
        }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        try {
            val currentTime = System.currentTimeMillis()
            // 精确控制50Hz采样率
            if (currentTime - lastSensorUpdateTime < SENSOR_INTERVAL_MS) return
            lastSensorUpdateTime = currentTime

            event?.let {
                when (it.sensor.type) {
                    Sensor.TYPE_ACCELEROMETER -> {
                        // 更新加速度计数据
                        System.arraycopy(it.values, 0, accelerometerData, 0, 3)
                        accelerometerTextView.text =
                            "加速度计\nX: ${formatFloat(it.values[0])}\nY: ${formatFloat(it.values[1])}\nZ: ${formatFloat(it.values[2])}"
                    }
                    Sensor.TYPE_GYROSCOPE -> {
                        // 更新陀螺仪数据
                        System.arraycopy(it.values, 0, gyroscopeData, 0, 3)
                        gyroscopeTextView.text =
                            "陀螺仪\nX: ${formatFloat(it.values[0])}\nY: ${formatFloat(it.values[1])}\nZ: ${formatFloat(it.values[2])}"
                    }
                }

                // 如果正在记录数据，添加到分类器
                if (isPredicting && ::motionClassifier.isInitialized && currentState != STATE_IDLE) {
                    motionClassifier.addData(
                        accelerometerData.copyOf(),
                        gyroscopeData.copyOf()
                    )
                }
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "传感器数据处理失败", e)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    /**
     * 切换记录状态
     */
    private fun toggleRecordingState(newState: String) {
        try {
            if (currentState == newState) {
                // 如果点击已激活状态的按钮，则停止记录
                stopRecording()
            } else {
                // 否则开始新的记录
                stopRecording() // 确保停止任何正在进行的记录
                startRecording(newState)
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "切换记录状态失败", e)
            Toast.makeText(this, "操作失败: ${e.message?.substring(0, 20) ?: "未知错误"}", Toast.LENGTH_SHORT).show()
        }
    }

    /**
     * 开始记录传感器数据
     */
    private fun startRecording(state: String) {
        predictionStartTime = SystemClock.elapsedRealtime()
        currentState = state
        isPredicting = true

        // 重置分类器缓冲区
        if (::motionClassifier.isInitialized) {
            motionClassifier.resetBuffer()
        }

        // 更新UI
        updateButtonStates()
        statusTextView.text = "正在记录: ${getStateName(state)}"
        resultTextView.text = "开始检测${getStateName(state)}状态..."
        predictionTextView.text = "正在采集数据..."
    }

    /**
     * 停止记录传感器数据并获取预测结果
     */
    private fun stopRecording() {
        if (currentState == STATE_IDLE) return

        // 检查采集时间是否足够
        val duration = SystemClock.elapsedRealtime() - predictionStartTime
        if (duration < MIN_COLLECTION_DURATION) {
            Toast.makeText(
                this,
                "采集时间太短 (${duration}ms < $MIN_COLLECTION_DURATION ms)，请至少采集${MIN_COLLECTION_DURATION / 1000}秒",
                Toast.LENGTH_SHORT
            ).show()
        } else if (::motionClassifier.isInitialized) {
            try {
                // 获取预测结果
                val prediction = motionClassifier.predictAll()

                prediction?.let { (label, confidence) ->
                    runOnUiThread {
                        predictionTextView.text =
                            "最终结果: ${translateLabel(label)}\n置信度: ${"%.1f".format(confidence * 100)}%"

                        resultTextView.text =
                            "检测到: ${translateLabel(label)} (${"%.1f%%".format(confidence * 100)})"

                        // 根据运动状态改变背景色
                        setBackgroundBasedOnState(label)
                    }
                }
            } catch (e: Exception) {
                Log.e("Prediction", "最终预测失败", e)
                runOnUiThread {
                    predictionTextView.text = "预测错误: ${e.message?.substring(0, 20)}"
                }
            }
        }

        val stoppedState = currentState
        currentState = STATE_IDLE
        isPredicting = false

        // 更新UI
        updateButtonStates()
        statusTextView.text = "${getStateName(stoppedState)}记录已停止"
    }

    /**
     * 更新按钮状态
     */
    private fun updateButtonStates() {
        try {
            val activeButton = when (currentState) {
                STATE_STATIC -> btnStatic
                STATE_WALKING -> btnWalking
                STATE_UPSTAIRS -> btnUpstairs
                STATE_DOWNSTAIRS -> btnDownstairs
                STATE_UPHILL -> btnUphill
                STATE_RUNNING -> btnRunning
                else -> null
            }

            // 重置所有按钮状态
            val allButtons = listOf(btnStatic, btnWalking, btnUpstairs, btnDownstairs, btnUphill, btnRunning)
            allButtons.forEach { button ->
                val baseText = button.text.split("\n")[0]
                button.text = "$baseText\n(未开始)"
                button.backgroundTintList = ContextCompat.getColorStateList(this, android.R.color.holo_blue_dark)
            }

            // 更新活动按钮状态
            activeButton?.let {
                it.text = "${it.text.split("\n")[0]}\n(记录中)"
                it.backgroundTintList = ContextCompat.getColorStateList(this, android.R.color.holo_green_dark)
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "更新按钮状态失败", e)
        }
    }

    /**
     * 获取状态名称
     */
    private fun getStateName(state: String): String {
        return when (state) {
            STATE_STATIC -> "静止"
            STATE_WALKING -> "步行"
            STATE_UPSTAIRS -> "上楼梯"
            STATE_DOWNSTAIRS -> "下楼梯"
            STATE_UPHILL -> "爬坡"
            STATE_RUNNING -> "跑步"
            else -> "未记录"
        }
    }

    /**
     * 格式化浮点数显示
     */
    private fun formatFloat(value: Float): String {
        return "%.3f".format(value)
    }

    /**
     * 翻译标签为中文
     */
    private fun translateLabel(label: String): String {
        return when (label) {
            "DownStairs" -> "下楼梯"
            "Running" -> "跑步"
            "Static" -> "静止"
            "Uphills" -> "爬坡"
            "Upstairs" -> "上楼梯"
            "Walking" -> "步行"
            else -> label
        }
    }

    /**
     * 根据状态设置背景色
     */
    private fun setBackgroundBasedOnState(state: String) {
        try {
            val color = when (state) {
                "Static" -> Color.rgb(70, 130, 180)    // 钢蓝色 - 静止
                "Walking" -> Color.rgb(50, 205, 50)     // 酸橙色 - 步行
                "Running" -> Color.rgb(220, 20, 60)     // 猩红色 - 跑步
                "Upstairs" -> Color.rgb(255, 215, 0)    // 金色 - 上楼梯
                "DownStairs" -> Color.rgb(147, 112, 219) // 中紫色 - 下楼梯
                "Uphills" -> Color.rgb(0, 191, 255)    // 深天蓝色 - 爬坡
                else -> Color.DKGRAY
            }
            predictionTextView.setBackgroundColor(color)
        } catch (e: Exception) {
            Log.e("MainActivity", "设置背景色失败", e)
        }
    }
}

class MotionClassifier(context: Context) {
    // 模型参数（与Python训练代码完全一致）
    private val WINDOW_SIZE = 100
    private val NUM_FEATURES = 8
    private val NUM_CLASSES = 6

    // 运动类别名称
    private val CLASS_NAMES = arrayOf(
        "DownStairs", "Running", "Static", "Uphills", "Upstairs", "Walking"
    )

    // TensorFlow Lite解释器
    private val tflite: InterpreterApi

    // 更新标准化参数（使用您提供的最新值）
    private val mean = floatArrayOf(
        8.201186579463636f,
        -1.8054629302274954f,
        2.0080382608744807f,
        0.28030321015503484f,
        0.2873924135594243f,
        0.29939560838949f,
        13.865204945640215f,
        2.0099370576999016f
    )

    private val std = floatArrayOf(
        7.9857884654221785f,
        7.379139596409335f,
        4.860793616294594f,
        2.7236888964845893f,
        2.8230227460744697f,
        3.380412873534198f,
        6.579527592140109f,
        1.9584149290133739f
    )

    // 数据缓冲区
    private val rawData = mutableListOf<FloatArray>()  // 存储标准化后的特征数据

    // 线程安全锁
    private val lock = ReentrantLock()

    init {
        try {
            Log.d("MotionClassifier", "开始加载模型文件...")
            val modelBuffer = loadModelFile(context, "motion_model.tflite")

            // 使用新API创建解释器
            val options = Options().apply {
                setNumThreads(4) // 设置线程数
            }

            tflite = InterpreterApi.create(modelBuffer, options)
            Log.d("MotionClassifier", "TensorFlow Lite解释器创建成功")
            Log.d("MotionClassifier", "标准化参数 - 均值: ${mean.contentToString()}, 标准差: ${std.contentToString()}")
        } catch (e: Exception) {
            Log.e("MotionClassifier", "初始化失败", e)
            throw RuntimeException("运动分类器初始化失败: ${e.message}")
        }
    }

    private fun loadModelFile(context: Context, modelName: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength

        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            startOffset,
            declaredLength
        ).apply {
            order(ByteOrder.nativeOrder())
        }
    }

    fun resetBuffer() {
        lock.lock()
        try {
            rawData.clear()
            Log.d("MotionClassifier", "缓冲区已重置")
        } finally {
            lock.unlock()
        }
    }

    /**
     * 添加传感器数据到缓冲区
     */
    fun addData(accelData: FloatArray, gyroData: FloatArray) {
        lock.lock()
        try {
            // 计算衍生特征（与Python训练代码完全一致）
            val accMag = sqrt(
                accelData[0] * accelData[0] +
                        accelData[1] * accelData[1] +
                        accelData[2] * accelData[2]
            )

            val gyroMag = sqrt(
                gyroData[0] * gyroData[0] +
                        gyroData[1] * gyroData[1] +
                        gyroData[2] * gyroData[2]
            )

            // 创建特征数组（8个特征）
            val features = floatArrayOf(
                accelData[0], accelData[1], accelData[2],
                gyroData[0], gyroData[1], gyroData[2],
                accMag, gyroMag
            )

            // 应用标准化（使用训练时的真实参数）
            val normalizedFeatures = FloatArray(NUM_FEATURES)
            for (i in 0 until NUM_FEATURES) {
                normalizedFeatures[i] = (features[i] - mean[i]) / std[i]
            }

            // 保存标准化后的特征
            rawData.add(normalizedFeatures)
        } catch (e: Exception) {
            Log.e("MotionClassifier", "添加数据失败", e)
        } finally {
            lock.unlock()
        }
    }

    /**
     * 对所有收集到的数据进行预测
     */
    fun predictAll(): Pair<String, Float>? {
        lock.lock()
        try {
            if (rawData.size < WINDOW_SIZE) {
                Log.w("MotionClassifier", "数据不足: ${rawData.size}/$WINDOW_SIZE")
                return null
            }

            // 准备用于预测的数据（精确匹配模型输入形状 [1, 100, 8]）
            val inputData = Array(1) { Array(WINDOW_SIZE) { FloatArray(NUM_FEATURES) } }

            // 获取最近的100个数据点
            val startIndex = rawData.size - WINDOW_SIZE
            for (i in 0 until WINDOW_SIZE) {
                val features = rawData[startIndex + i]
                System.arraycopy(features, 0, inputData[0][i], 0, NUM_FEATURES)
            }

            // 运行模型
            val output = Array(1) { FloatArray(NUM_CLASSES) }
            tflite.run(inputData, output)

            // 获取预测结果
            val probabilities = output[0]
            val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
            val prediction = CLASS_NAMES[maxIndex]
            val confidence = probabilities[maxIndex]

            Log.d("MotionClassifier", "预测结果: $prediction (${"%.1f%%".format(confidence * 100)}), 概率分布: ${probabilities.contentToString()}")

            return Pair(prediction, confidence)
        } catch (e: Exception) {
            Log.e("MotionClassifier", "预测失败", e)
            return null
        } finally {
            lock.unlock()
        }
    }

    fun close() {
        try {
            tflite.close()
            Log.d("MotionClassifier", "模型已关闭")
        } catch (e: Exception) {
            Log.e("MotionClassifier", "关闭模型失败", e)
        }
    }
}