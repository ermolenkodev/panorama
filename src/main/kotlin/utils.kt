import java.io.File

fun projectRoot(): String {
    val directory = File("");
    return directory.absolutePath
}
