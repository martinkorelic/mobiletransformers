package com.martinkorelic.mobiletransformers

import io.pebbletemplates.pebble.error.PebbleException
import io.pebbletemplates.pebble.extension.AbstractExtension
import io.pebbletemplates.pebble.extension.Function
import io.pebbletemplates.pebble.template.EvaluationContext
import io.pebbletemplates.pebble.template.PebbleTemplate
import io.pebbletemplates.pebble.operator.BinaryOperator
import io.pebbletemplates.pebble.operator.BinaryOperatorType
import io.pebbletemplates.pebble.operator.Associativity
import io.pebbletemplates.pebble.node.expression.BinaryExpression
import io.pebbletemplates.pebble.template.EvaluationContextImpl
import io.pebbletemplates.pebble.template.PebbleTemplateImpl
import java.util.Objects

class RaiseExceptionFunction : Function {

    override fun getArgumentNames(): MutableList<String> {
        return mutableListOf("message");
    }

    override fun execute(
        args: Map<String?, Any?>,
        self: PebbleTemplate,
        context: EvaluationContext?,
        lineNumber: Int
    ): Any {
        val msg = if (args["message"] != null) args["message"].toString() else "Unknown error"

        // You can throw a PebbleException to include context and line number
        throw PebbleException(null, msg, lineNumber, self.name)
    }

    val name: String
        get() = "raise_exception"
}

class ChatTemplatePebbleExtension : AbstractExtension() {
    override fun getFunctions(): Map<String, Function> {
        val functions: MutableMap<String, Function> = HashMap()
        functions["raise_exception"] = RaiseExceptionFunction()
        return functions
    }
}

class TypeSafeComparisonExtension : AbstractExtension() {

    override fun getBinaryOperators(): List<BinaryOperator> {
        return listOf(
            TypeSafeNotEqualsOperator(),
            TypeSafeEqualsOperator()
        )
    }

    // Custom neq operator (not equals) - use 'neq' instead of '!='
    class TypeSafeNotEqualsOperator : BinaryOperator {

        override fun getPrecedence(): Int = 30
        override fun getSymbol(): String = "neq"  // Use 'neq' instead of '!='
        override fun createInstance(): BinaryExpression<*> = TypeSafeNotEqualsExpression()
        override fun getType(): BinaryOperatorType = BinaryOperatorType.NORMAL
        override fun getAssociativity(): Associativity = Associativity.LEFT
    }

    // Custom eq operator (equals) - use 'eq' instead of '=='
    class TypeSafeEqualsOperator : BinaryOperator {

        override fun getPrecedence(): Int = 30
        override fun getSymbol(): String = "eq"   // Use 'eq' instead of '=='
        override fun createInstance(): BinaryExpression<*> = TypeSafeEqualsExpression()
        override fun getType(): BinaryOperatorType = BinaryOperatorType.NORMAL
        override fun getAssociativity(): Associativity = Associativity.LEFT
    }

    // Expression implementation for neq
    class TypeSafeNotEqualsExpression : BinaryExpression<Any>() {

        override fun evaluate(self: PebbleTemplateImpl?, context: EvaluationContextImpl?): Any {
            val left = leftExpression.evaluate(self, context)
            val right = rightExpression.evaluate(self, context)

            return !isEqual(left, right)
        }
    }

    // Expression implementation for eq
    class TypeSafeEqualsExpression : BinaryExpression<Any>() {

        override fun evaluate(self: PebbleTemplateImpl?, context: EvaluationContextImpl?): Any {
            val left = leftExpression.evaluate(self, context)
            val right = rightExpression.evaluate(self, context)

            return isEqual(left, right)
        }
    }

    companion object {
        // Helper method for type-safe equality comparison
        fun isEqual(left: Any?, right: Any?): Boolean {
            // Handle null cases
            if (left == null && right == null) return true
            if (left == null || right == null) return false

            // If both are the same type, use Objects.equals
            if (left.javaClass == right.javaClass) {
                return Objects.equals(left, right)
            }

            // Handle numeric comparisons
            if (isNumeric(left) && isNumeric(right)) {
                return compareNumbers(left, right) == 0
            }

            // Handle boolean comparisons with type coercion
            if (left is Boolean || right is Boolean) {
                return toBooleanValue(left) == toBooleanValue(right)
            }

            // Handle string comparisons
            if (left is String || right is String) {
                return left.toString() == right.toString()
            }

            // Fallback to Objects.equals
            return Objects.equals(left, right)
        }

        // Helper method to check if object is numeric
        private fun isNumeric(obj: Any?): Boolean = obj is Number

        // Helper method to compare numbers of different types
        private fun compareNumbers(left: Any, right: Any): Int {
            return when {
                left is Double || right is Double ->
                    (left as Number).toDouble().compareTo((right as Number).toDouble())
                left is Float || right is Float ->
                    (left as Number).toFloat().compareTo((right as Number).toFloat())
                left is Long || right is Long ->
                    (left as Number).toLong().compareTo((right as Number).toLong())
                else ->
                    (left as Number).toInt().compareTo((right as Number).toInt())
            }
        }

        // Helper method to convert objects to boolean values
        private fun toBooleanValue(obj: Any?): Boolean {
            return when (obj) {
                is Boolean -> obj
                is Number -> obj.toDouble() != 0.0
                is String -> obj.isNotEmpty() && !obj.equals("false", ignoreCase = true) && obj != "0"
                else -> obj != null
            }
        }
    }
}