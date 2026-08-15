package com.martinkorelic.mobiletransformers

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * [Tasks.TASKS] and `getPreprocessFunctionForTask`'s dispatch must name the same set.
 *
 * The list exists so a UI can offer a picker instead of asking the user to spell a task name from
 * memory — `DatasetConfig.task` is a free string, and a typo surfaces as `Unsupported task: …`
 * minutes into a training run. A list that drifts from the dispatch reintroduces exactly that failure
 * while looking like it prevents it: the picker would offer a name the trainer then rejects.
 *
 * Checked in both directions, because each direction is a different mistake. A dispatch entry missing
 * from the list is a task nobody can select; a list entry missing from the dispatch is an option that
 * fails when chosen.
 */
class TaskRegistryTest {

    @Test
    fun everyListedTaskResolvesToAPreprocessor() {
        for (task in Tasks.TASKS) {
            val preprocessor = getPreprocessFunctionForTask(task.name)
            assertNotNull("'${task.name}' is offered but has no preprocessor", preprocessor)
        }
    }

    /**
     * The reverse direction, read off the source rather than the dispatch — there is no way to
     * enumerate a `when`'s branches at runtime, and a test that could only see the list would pass
     * vacuously when a preprocessor was added without listing it.
     */
    @Test
    fun everyDispatchBranchIsListed() {
        val source = TestSources.read("main/java/com/martinkorelic/mobiletransformers/DataUtil.kt")
        val dispatchBody = source
            .substringAfter("fun getPreprocessFunctionForTask")
            .substringAfter("return when (taskName.lowercase()) {")
            .substringBefore("else ->")
        val branches = Regex("\"([a-z_]+)\"\\s*->").findAll(dispatchBody).map { it.groupValues[1] }.toList()

        assertTrue("could not read the dispatch branches — the guard would pass vacuously", branches.size >= 5)
        assertEquals(
            "the task list and the dispatch have drifted",
            branches.sorted(),
            Tasks.NAMES.sorted(),
        )
    }

    @Test
    fun everyListedTaskDescribesWhatItsRowsContain() {
        for (task in Tasks.TASKS) {
            assertTrue("'${task.name}' has no description", task.description.isNotBlank())
            assertEquals(task.description, Tasks.describe(task.name))
        }
    }

    @Test
    fun anUnlistedTaskStillFailsClosed() {
        // The picker removes the typo; it must not remove the check behind it.
        assertThrows(IllegalArgumentException::class.java) {
            getPreprocessFunctionForTask("mobileactions")
        }
    }
}
