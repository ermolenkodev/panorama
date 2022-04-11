sealed class Output<T> {
    class Success<T>(public val data: T) : Output<T>()
    class Failure<T>(public val msg: String, public val e: Exception? = null) : Output<T>()
}
