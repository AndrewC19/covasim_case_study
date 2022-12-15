file=$1
fixed=$2
cat "$file" | jq -r 'keys[]' |
while IFS= read -r location; do
    variant=$(jq -r ".${location}.variant" $file)
    seed=$(jq -r ".${location}.seed" $file)
    echo "Simulating location $location with dominant variant $variant and seed $seed..."
    sbatch simulate_location.sh "$location" "$variant" $seed 30 $fixed
done