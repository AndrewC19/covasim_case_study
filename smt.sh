file=$1
cat "$file" | jq -r 'keys[]' |
while IFS= read -r location; do
    variant_a=$(jq -r ".${location}.variants[0]" $file)
    seed_a=$(jq -r ".${location}.seeds[0]" $file)
    echo "Simulating location $location with dominant variant $variant_a and seed $seed_a..."
    sbatch simulate_location.sh "$location" "$variant_a" $seed_a 30 True
    variant_b=$(jq -r ".${location}.variants[1]" $file)
    seed_b=$(jq -r ".${location}.seeds[1]" $file)
    echo "Simulating location $location with dominant variant $variant_b and seed $seed_b..."
    sbatch simulate_location.sh "$location" "$variant_b" $seed_b 30 True
done