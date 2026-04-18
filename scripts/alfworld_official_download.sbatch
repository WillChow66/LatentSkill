#!/bin/bash
#SBATCH --job-name=alfworld_dl
#SBATCH --account=yzhao010_1245
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/alfworld_dl_%j.out
#SBATCH --error=logs/alfworld_dl_%j.err

source /home1/xzhou733/latentskill/scripts/setup_env.sh

# Download to a separate directory for comparison
export ALFWORLD_DATA="/home1/xzhou733/latentskill/alfworld_official"
mkdir -p $ALFWORLD_DATA

echo "Running alfworld-download..."
alfworld-download -f

echo ""
echo "=== Downloaded structure ==="
ls -la $ALFWORLD_DATA/
echo ""
echo "=== Game count comparison ==="
for split in train valid_train valid_seen valid_unseen; do
    if [ -d "$ALFWORLD_DATA/json_2.1.1/$split" ]; then
        count=$(find $ALFWORLD_DATA/json_2.1.1/$split -name "game.tw-pddl" 2>/dev/null | wc -l)
        echo "Official $split: $count games"
    fi
done
echo ""
echo "Compare with our existing data:"
for split in train valid_train valid_seen valid_unseen; do
    count=$(find /home1/xzhou733/latentskill/SkillRL/agent_system/environments/env_package/alfworld/json_2.1.1/$split -name "game.tw-pddl" 2>/dev/null | wc -l)
    echo "Existing $split: $count games"
done
