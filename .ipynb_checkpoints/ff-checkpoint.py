def fibonacci(n):
    """Generate Fibonacci sequence up to n."""
    sequence = []
    a, b = 0, 1
    while a < n:
        sequence.append(a)
        a, b = b, a + b
    return sequence

# Get user input
try:
    limit = int(input("Enter a number to generate Fibonacci sequence up to: "))
    if limit < 0:
        print("Please enter a positive integer.")
    else:
        fib_sequence = fibonacci(limit)
        print(f"Fibonacci sequence up to {limit}: {fib_sequence}")
except ValueError:
    print("Invalid input! Please enter an integer.")
