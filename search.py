from query_processor import process_query, boolean_and

def search(query):
    # TODO:
    # Process query
    # Retrieve postings
    # Boolean AND
    # Rank documents
    # Convert document IDs to URLs
    pass


def main():
    print("UCI Search Engine")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Search> ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        search(query)


if __name__ == "__main__":
    main()