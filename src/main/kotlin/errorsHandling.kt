sealed class Output<T> {
    class Success<T>(val data: T) : Output<T>()
    class Failure<T>(val msg: String, val e: Exception? = null) : Output<T>()
}
